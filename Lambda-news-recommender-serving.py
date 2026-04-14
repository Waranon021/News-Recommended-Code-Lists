# This code goes inside the AWS Lambda Function code editor, not in VS Code terminal.

import json                                                                                 # Import json so the Lambda function can read request bodies and return JSON responses.
import os                                                                                   # Import os so the function can read optional environment variables for bucket and table names.
import pickle                                                                               # Import pickle so the function can load the trained model file from S3.
from decimal import Decimal                                                                 # Import Decimal so DynamoDB numeric values can be converted safely into JSON-friendly values.

import boto3                                                                                # Import boto3 so the function can talk to S3 and DynamoDB.


MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "news-recommending-models-st125934")         # Read the models bucket name from Lambda environment variables, or use current bucket as default.
MODEL_KEY = os.environ.get("MODEL_KEY", "registry/latest_model.pkl")                        # Read the latest model key from environment variables, or use registry/latest_model.pkl by default.
USER_TABLE_NAME = os.environ.get("USER_TABLE_NAME", "user-vectors")                         # Read the DynamoDB user table name from environment variables, or use user-vectors by default.
ARTICLE_TABLE_NAME = os.environ.get("ARTICLE_TABLE_NAME", "article-embeddings")             # Read the DynamoDB article table name from environment variables, or use article-embeddings by default.
DEFAULT_K = 10                                                                              # Use 10 recommendations by default when k is not provided in the request.
MAX_K = 20                                                                                  # Cap the maximum requested recommendation count to keep Lambda responses manageable.


s3 = boto3.client("s3")                                                                     # Create one S3 client that will download the latest trained model from the models bucket.
dynamodb_resource = boto3.resource("dynamodb")                                              # Create one DynamoDB resource used to access the two DynamoDB tables.
dynamodb_client = boto3.client("dynamodb")                                                  # Create one DynamoDB client used for batch_get_item when loading many article details at once.

user_table = dynamodb_resource.Table(USER_TABLE_NAME)                                       # Get the DynamoDB table object for user-vectors.
article_table = dynamodb_resource.Table(ARTICLE_TABLE_NAME)                                 # Get the DynamoDB table object for article-embeddings.


MODEL_CACHE = None                                                                          # Create a global model cache so Lambda can reuse the loaded model across warm invocations.
MODEL_ETAG = None                                                                           # Create a global ETag cache so Lambda can detect whether the S3 model file has changed.


def decimal_to_native(value):                                                               # Define a helper function that converts DynamoDB Decimal values into normal Python types.
    if isinstance(value, list):                                                             # Check whether the value is a list.
        return [decimal_to_native(x) for x in value]                                        # Convert every element in the list recursively.
    if isinstance(value, dict):                                                             # Check whether the value is a dictionary.
        return {k: decimal_to_native(v) for k, v in value.items()}                          # Convert every value in the dictionary recursively.
    if isinstance(value, Decimal):                                                          # Check whether the value is a DynamoDB Decimal number.
        if value % 1 == 0:                                                                  # Check whether the Decimal number is mathematically an integer.
            return int(value)                                                               # Return it as normal Python int when it has no fractional part.
        return float(value)                                                                 # Otherwise return it as a normal Python float.
    return value                                                                            # Return the value unchanged when no conversion is needed.


def build_response(status_code, payload):                                                   # Define a helper function that returns standard Lambda HTTP-style responses.
    return {                                                                                # Return a response dictionary in the format Lambda proxy integrations expect.
        "statusCode": status_code,                                                          # Set the HTTP-style status code such as 200 or 400.
        "headers": {"Content-Type": "application/json"},                                    # Set the response content type to JSON.
        "body": json.dumps(payload, default=decimal_to_native)                              # Convert the payload dictionary into JSON text, handling Decimal values safely.
    }


def parse_event_body(event):                                                                # Define a helper function that safely extracts the request body from Lambda event input.
    if "body" not in event or event["body"] is None:                                        # Check whether the event has no body field at all.
        return event                                                                        # Treat the entire event as the payload when using direct test events in Lambda.
    if isinstance(event["body"], str):                                                      # Check whether the body is a JSON string, which is common with API Gateway.
        return json.loads(event["body"])                                                    # Parse the JSON string into a normal Python dictionary.
    return event["body"]                                                                    # Otherwise return the body directly when it is already a dictionary.


def fill_with_popular(base_recs, seen_set, popular_articles, k):                            # Define a helper function that fills short recommendation lists using popular fallback items.
    final_recs = []                                                                         # Create an empty list that will hold the final recommendation IDs.
    used = set()                                                                            # Create a set to prevent duplicate recommendation IDs.

    for article_id in base_recs:                                                            # Loop through the first-stage recommendation candidates.
        if article_id not in seen_set and article_id not in used:                           # Keep only unseen and not-yet-added items.
            final_recs.append(article_id)                                                   # Add the candidate article into the final list.
            used.add(article_id)                                                            # Mark the candidate as already used.
        if len(final_recs) >= k:                                                            # Stop early when the recommendation list already reached the target size.
            return final_recs[:k]                                                           # Return the first k results immediately.

    for article_id in popular_articles:                                                     # If the list is still too short, top it up using popular fallback items.
        if article_id not in seen_set and article_id not in used:                           # Again keep only unseen and unique items.
            final_recs.append(article_id)                                                   # Add the fallback article into the final list.
            used.add(article_id)                                                            # Mark the fallback article as already used.
        if len(final_recs) >= k:                                                            # Stop once the list reaches the requested size.
            break                                                                           # Break out of the fallback loop.

    return final_recs[:k]                                                                   # Return the final filled recommendation list limited to k items.


def load_model():                                                                           # Define a helper function that downloads and caches the latest model from S3.
    global MODEL_CACHE                                                                      # Tell Python that this function should update the global model cache variable.
    global MODEL_ETAG                                                                       # Tell Python that this function should update the global ETag cache variable.

    head = s3.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)                               # Read the S3 object metadata for the latest model file.
    current_etag = head["ETag"]                                                             # Extract the current S3 ETag so it can detect model file changes.

    if MODEL_CACHE is not None and MODEL_ETAG == current_etag:                              # Check whether the model is already cached and still matches the current S3 file.
        return MODEL_CACHE                                                                  # Reuse the cached model to avoid downloading and unpickling again.

    local_path = "/tmp/latest_model.pkl"                                                    # Choose a temporary Lambda local path where the model file will be downloaded.
    s3.download_file(MODEL_BUCKET, MODEL_KEY, local_path)                                   # Download the latest model pickle from S3 into Lambda temporary storage.

    with open(local_path, "rb") as f:                                                       # Open the downloaded pickle file in binary read mode.
        MODEL_CACHE = pickle.load(f)                                                        # Load the model object from the pickle file into memory.

    MODEL_ETAG = current_etag                                                               # Save the latest S3 ETag into the global cache state.
    return MODEL_CACHE                                                                      # Return the freshly loaded model object.


def get_user_profile(user_id):                                                              # Define a helper function that reads one user profile from DynamoDB.
    response = user_table.get_item(Key={"user_id": str(user_id)})                           # Query the user-vectors table using user_id as the partition key.
    item = response.get("Item", {})                                                         # Extract the DynamoDB item if it exists, or use empty dictionary if missing.
    return decimal_to_native(item)                                                          # Convert any Decimal values into normal Python values before returning.


def get_article_details(article_ids):                                                       # Define a helper function that batch-loads article metadata from DynamoDB.
    if not article_ids:                                                                     # Check whether the requested article ID list is empty.
        return []                                                                           # Return an empty list immediately when there is nothing to fetch.

    keys = [{"article_id": {"S": str(article_id)}} for article_id in article_ids[:100]]     # Build DynamoDB batch-get keys for up to the first 100 requested article IDs.
    response = dynamodb_client.batch_get_item(                                              # Call batch_get_item so many article rows can be retrieved in one request.
        RequestItems={                                                                      # Build the request payload for the batch read.
            ARTICLE_TABLE_NAME: {                                                           # Use the article-embeddings table as the source table.
                "Keys": keys                                                                # Pass the list of article_id keys to fetch.
            }
        }
    )

    items = response.get("Responses", {}).get(ARTICLE_TABLE_NAME, [])                       # Extract the raw returned items list from the batch_get_item response.
    article_map = {}                                                                        # Create a dictionary so returned items can be reordered to match the recommendation list.
    for item in items:                                                                      # Loop through each returned DynamoDB raw item.
        article_id = item["article_id"]["S"]                                                # Extract the article_id string from the DynamoDB raw format.
        article_map[article_id] = {                                                         # Convert the raw DynamoDB item into a cleaner response object.
            "article_id": article_id,                                                       # Save article_id.
            "title": item.get("title", {}).get("S", ""),                                    # Save title with empty-string fallback.
            "url": item.get("url", {}).get("S", ""),                                        # Save url with empty-string fallback.
            "category": item.get("category", {}).get("S", ""),                              # Save category with empty-string fallback.
            "source": item.get("source", {}).get("S", ""),                                  # Save source with empty-string fallback.
            "abstract": item.get("abstract", {}).get("S", ""),                              # Save abstract with empty-string fallback.
            "pub_date": item.get("pub_date", {}).get("S", ""),                              # Save pub_date with empty-string fallback.
            "data_source": item.get("data_source", {}).get("S", ""),                        # Save data_source with empty-string fallback.
        }

    ordered = []                                                                            # Create a list that will preserve the original recommendation order.
    for article_id in article_ids:                                                          # Loop through the recommendation IDs in their original order.
        if article_id in article_map:                                                       # Keep only IDs that were actually found in DynamoDB.
            ordered.append(article_map[article_id])                                         # Add the cleaned article metadata into the ordered results list.

    return ordered                                                                          # Return the ordered article detail list.


def recommend_popularity(user_profile, model, k):                                           # Define the popularity-baseline recommender used during serving.
    seen_set = set(user_profile.get("recent_clicks", []))                                   # Build the set of already clicked articles from the user profile.
    popular_articles = model.get("popular_articles", [])                                     # Read the popularity ranking stored in the saved model object.
    return fill_with_popular([], seen_set, popular_articles, k)                             # Return top-k popular unseen articles.


def recommend_svd(user_profile, model, k):                                                  # Define the SVD recommender used during serving.
    seen_set = set(user_profile.get("recent_clicks", []))                                   # Build the set of already clicked articles from the user profile.
    history = user_profile.get("recent_clicks", [])                                         # Read the user's recent clicked article list from the profile.
    if not history:                                                                         # Check whether the user has no recent clicks at all.
        return fill_with_popular([], seen_set, model.get("popular_articles", []), k)        # Fall back to popularity when no user history exists.

    article_index = model["article_index"]                                                  # Read the article_id -> row index mapping stored in the SVD model.
    article_ids = model["article_ids"]                                                      # Read the ordered article ID list stored in the SVD model.
    item_factors = model["item_factors"]                                                    # Read the learned latent item factor matrix from the SVD model.
    knn_model = model["knn_model"]                                                          # Read the fitted nearest-neighbor model from the SVD model.
    popular_articles = model.get("popular_articles", [])                                    # Read the popularity fallback list from the SVD model.
    max_seed_items = int(model.get("max_seed_items", 3))                                    # Read the maximum number of recent seed items used by this model.

    seed_items = [item for item in history[-max_seed_items:] if item in article_index]      # Keep only the user's last few clicked items that exist in the SVD article index.
    if not seed_items:                                                                      # Check whether no usable seed item exists.
        return fill_with_popular([], seen_set, popular_articles, k)                         # Fall back to popularity when none of the user's seeds are usable.

    scores = {}                                                                             # Create a score dictionary to accumulate recommendation strength across seed items.
    n_neighbors = min(50, len(article_ids))                                                 # Choose a safe neighbor count based on the size of the article universe.

    for seed_item in seed_items:                                                            # Loop through each usable seed item.
        seed_idx = article_index[seed_item]                                                 # Convert the seed article ID into its SVD latent-factor row index.
        distances, indices = knn_model.kneighbors(item_factors[[seed_idx]], n_neighbors=n_neighbors)  # Retrieve the nearest latent-space neighbors for that seed item.

        for dist, idx in zip(distances[0], indices[0]):                                     # Loop through each returned neighbor candidate.
            candidate = article_ids[idx]                                                    # Convert the neighbor row index back into article_id.
            if candidate == seed_item or candidate in seen_set:                             # Skip the seed item itself and anything the user has already clicked.
                continue                                                                    # Continue to the next candidate.
            scores[candidate] = scores.get(candidate, 0.0) + (1.0 - float(dist))           # Accumulate similarity-based score using cosine similarity proxy.

    ranked = [article_id for article_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]  # Sort candidate IDs by descending accumulated score.
    return fill_with_popular(ranked, seen_set, popular_articles, k)                         # Fill any missing slots with popularity fallback articles.


def recommend_knn(user_profile, model, k):                                                  # Define the KNN content-based recommender used during serving.
    seen_set = set(user_profile.get("recent_clicks", []))                                   # Build the set of already clicked articles from the user profile.
    history = user_profile.get("recent_clicks", [])                                         # Read the user's recent clicked article list from the profile.
    if not history:                                                                         # Check whether the user has no recent clicks at all.
        return fill_with_popular([], seen_set, model.get("popular_articles", []), k)        # Fall back to popularity when no user history exists.

    article_index = model["article_index"]                                                  # Read the article_id -> row index mapping stored in the KNN model.
    article_ids = model["article_ids"]                                                      # Read the ordered article ID list stored in the KNN model.
    tfidf_matrix = model["tfidf_matrix"]                                                    # Read the saved TF-IDF matrix from the KNN model.
    knn_model = model["knn_model"]                                                          # Read the fitted nearest-neighbor model from the KNN model.
    popular_articles = model.get("popular_articles", [])                                    # Read the popularity fallback list from the KNN model.
    max_seed_items = int(model.get("max_seed_items", 3))                                    # Read the maximum number of recent seed items used by this model.

    seed_items = [item for item in history[-max_seed_items:] if item in article_index]      # Keep only the user's last few clicked items that exist in the KNN article index.
    if not seed_items:                                                                      # Check whether no usable seed item exists.
        return fill_with_popular([], seen_set, popular_articles, k)                         # Fall back to popularity when none of the user's seeds are usable.

    scores = {}                                                                             # Create a score dictionary to accumulate recommendation strength across seed items.
    n_neighbors = min(50, len(article_ids))                                                 # Choose a safe neighbor count based on the size of the article universe.

    for seed_item in seed_items:                                                            # Loop through each usable seed item.
        seed_idx = article_index[seed_item]                                                 # Convert the seed article ID into its TF-IDF row index.
        distances, indices = knn_model.kneighbors(tfidf_matrix[seed_idx], n_neighbors=n_neighbors)  # Retrieve the nearest text neighbors for that seed item.

        for dist, idx in zip(distances[0], indices[0]):                                     # Loop through each returned neighbor candidate.
            candidate = article_ids[idx]                                                    # Convert the neighbor row index back into article_id.
            if candidate == seed_item or candidate in seen_set:                             # Skip the seed item itself and anything the user has already clicked.
                continue                                                                    # Continue to the next candidate.
            scores[candidate] = scores.get(candidate, 0.0) + (1.0 - float(dist))           # Accumulate similarity-based score using cosine similarity proxy.

    ranked = [article_id for article_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]  # Sort candidate IDs by descending accumulated score.
    return fill_with_popular(ranked, seen_set, popular_articles, k)                         # Fill any missing slots with popularity fallback articles.


def lambda_handler(event, context):                                                          # Define the main Lambda handler function that AWS will call when the API request arrives.
    try:                                                                                     # Start a try block so the function can return clean error messages instead of crashing hard.
        payload = parse_event_body(event)                                                    # Parse the incoming Lambda event into a normal payload dictionary.
        user_id = str(payload.get("user_id", "")).strip()                                    # Read user_id from the request payload and force it into clean string form.
        k = int(payload.get("k", DEFAULT_K))                                                 # Read the requested recommendation count, or use the default when missing.
        k = max(1, min(k, MAX_K))                                                            # Keep k within the allowed range from 1 to MAX_K.

        if user_id == "":                                                                    # Check whether user_id was missing or blank.
            return build_response(400, {"error": "user_id is required"})                     # Return a clear client error if user_id is not provided.

        model = load_model()                                                                 # Load the latest trained model from S3, using cache when possible.
        model_type = model.get("model_type", "unknown")                                      # Read the model type stored inside the loaded model object.
        user_profile = get_user_profile(user_id)                                             # Load the user's profile from the user-vectors DynamoDB table.

        if model_type == "popularity_baseline":                                              # Check whether the loaded model is the popularity baseline.
            recommended_ids = recommend_popularity(user_profile, model, k)                   # Generate recommendations using the popularity recommender.
        elif model_type == "svd_matrix_factorisation":                                       # Check whether the loaded model is the SVD recommender.
            recommended_ids = recommend_svd(user_profile, model, k)                          # Generate recommendations using the SVD recommender.
        elif model_type == "knn_content_based":                                              # Check whether the loaded model is the KNN content-based recommender.
            recommended_ids = recommend_knn(user_profile, model, k)                          # Generate recommendations using the KNN content-based recommender.
        else:                                                                                # Handle unknown model types safely.
            return build_response(500, {"error": f"Unsupported model_type: {model_type}"})   # Return a server error if the loaded model type is not supported.

        recommendations = get_article_details(recommended_ids)                               # Batch-load article metadata for the recommended article IDs from DynamoDB.

        return build_response(200, {                                                         # Return the final successful recommendation response.
            "user_id": user_id,                                                              # Include the requested user_id in the response.
            "model_type": model_type,                                                        # Include the loaded model type in the response.
            "requested_k": k,                                                                # Include the requested number of recommendations.
            "total": len(recommendations),                                                   # Include how many article recommendations were actually returned.
            "recommendations": recommendations                                               # Include the final list of recommended article metadata objects.
        })

    except Exception as e:                                                                   # Catch any unexpected error during serving.
        return build_response(500, {"error": str(e)})                                        # Return the error as JSON so debugging is easier during testing.