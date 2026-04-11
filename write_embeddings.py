# write_embeddings.py

import os                                                                 # Import os for building local temporary file paths safely.
import tempfile                                                           # Import tempfile so parquet files can be downloaded into a temporary folder on Windows.
import hashlib                                                            # Import hashlib so the script can create stable article_id values from URL or fallback text.
import boto3                                                              # Import boto3 so the script can read parquet files from S3 and write embeddings into DynamoDB.
import pandas as pd                                                       # Import pandas so the script can load, combine, clean, and deduplicate parquet data.
from decimal import Decimal                                               # Import Decimal because DynamoDB stores numeric values more safely in Decimal format.
from sentence_transformers import SentenceTransformer                     # Import SentenceTransformer so the script can generate text embeddings for each article.


PROC_BUCKET = "news-recommending-processed-st125934"                      # Name of the processed S3 bucket that stores the article parquet files.
ARTICLES_PREFIX = "articles/"                                             # Folder inside the processed bucket where article parquet files are stored.
DYNAMO_TABLE = "article-embeddings"                                       # Name of the DynamoDB table where article embeddings will be saved.
AWS_REGION = "ap-southeast-7"                                             # AWS region for your current account, which is Thailand here.
MAX_ARTICLES = 5000                                                       # Maximum number of unique articles to embed in one run so runtime stays manageable.
MODEL_NAME = "all-MiniLM-L6-v2"                                           # Sentence-transformer model used to generate the embedding vectors.


s3 = boto3.client("s3", region_name=AWS_REGION)                           # Create one S3 client that all helper functions will use to read parquet files from S3.
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)             # Create one DynamoDB resource that will be used to write items into the target table.
table = dynamodb.Table(DYNAMO_TABLE)                                      # Get the specific DynamoDB table object for article embeddings.


def to_decimal(value):                                                    # Define a helper function to convert normal float values into DynamoDB-safe Decimal values.
    return Decimal(str(round(float(value), 6)))                           # Round each float slightly, convert it to string, then wrap it in Decimal.


def make_article_id(row):                                                 # Define a helper function that creates a stable article_id for each article row.
    url = str(row.get("url", "")).strip()                                 # Try to get the article URL first because URL is the best identifier in this dataset.

    if url:                                                               # Check whether the URL exists and is not empty.
        return hashlib.sha256(url.encode("utf-8")).hexdigest()            # Hash the URL into a stable SHA-256 article_id.

    fallback_text = "||".join([                                           # Build fallback text if URL is missing so the article can still get a stable ID.
        str(row.get("title", "")).strip(),                                # Use title as part of the fallback identifier.
        str(row.get("source_name", row.get("source", ""))).strip(),       # Use source or source_name weas part of the fallback identifier.
        str(row.get("publishedAt", row.get("pub_date", ""))).strip(),     # Use publishedAt or pub_date as part of the fallback identifier.
    ])

    return hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()      # Hash the fallback text into a stable SHA-256 article_id.


def list_parquet_files(bucket, prefix):                                   # Define a helper function that lists every parquet file under a given S3 folder prefix.
    paginator = s3.get_paginator("list_objects_v2")                       # Create a paginator because one S3 folder may contain many parquet files.
    parquet_keys = []                                                     # Create an empty list to store the matching parquet file keys.

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):         # Loop through every result page returned by S3 for the selected bucket and prefix.
        for obj in page.get("Contents", []):                              # Loop through each file object inside the current result page.
            if obj["Key"].endswith(".parquet"):                           # Keep only files that end with .parquet because those are the processed datasets we need.
                parquet_keys.append(obj["Key"])                           # Add the parquet file path into the list.

    return parquet_keys                                                   # Return the final list of parquet file paths.


def load_articles_from_s3(bucket, prefix):                                # Define a helper function that downloads all article parquet files from S3 and combines them.
    parquet_files = list_parquet_files(bucket, prefix)                    # Call the previous helper to get every parquet file under the articles/ folder.

    if not parquet_files:                                                 # Check whether no parquet files were found at all.
        raise ValueError(f"No parquet files found in s3://{bucket}/{prefix}")  # Stop with a clear error if the folder is empty.

    print(f"Found {len(parquet_files)} parquet files in s3://{bucket}/{prefix}")  # Print how many parquet parts were found in S3.

    tmp_dir = tempfile.gettempdir()                                       # Get the system temporary folder path so the files can be downloaded locally first.
    dfs = []                                                              # Create an empty list to hold each parquet part after loading it into pandas.

    for key in parquet_files:                                             # Loop through each parquet file path found in S3.
        local_path = os.path.join(tmp_dir, os.path.basename(key))         # Build a local temporary file path using only the parquet file name.
        s3.download_file(bucket, key, local_path)                         # Download the parquet file from S3 to the temporary local path.
        df = pd.read_parquet(local_path)                                  # Read the downloaded parquet file into a pandas DataFrame.
        dfs.append(df)                                                    # Add that dataframe into the list of parquet parts.

    articles_df = pd.concat(dfs, ignore_index=True)                       # Combine all parquet parts into one full article dataframe and reset row numbering.
    print(f"Loaded {len(articles_df)} raw article rows from parquet")     # Print the total number of raw rows before any cleaning.

    return articles_df                                                    # Return the combined raw article dataframe.


def build_text(row):                                                      # Define a helper function that builds the text used for generating the embedding.
    title = str(row.get("title", "")).strip()                             # Safely extract the title from the row and clean whitespace.
    abstract = str(row.get("abstract", "")).strip()                       # Safely extract the abstract from the row and clean whitespace.
    description = str(row.get("description", "")).strip()                 # Safely extract the description if the dataframe uses that column instead.
    content = str(row.get("content", "")).strip()                         # Safely extract the content from the row and clean whitespace.

    body = abstract or description or content                             # Choose the richest available field after title, preferring abstract first.
    return f"{title} {body}".strip()                                      # Combine title and body into one text string and remove extra whitespace.


print(f"Loading model: {MODEL_NAME}")                                     # Print the model name to know which embedding model is being loaded.
model = SentenceTransformer(MODEL_NAME)                                   # Load the sentence-transformer model into memory.
print("Model loaded successfully")                                        # Print a success message after the model finishes loading.

articles_df = load_articles_from_s3(PROC_BUCKET, ARTICLES_PREFIX)         # Load all article parquet files from S3 into one combined dataframe.

articles_df = articles_df[                                                # Keep only rows where title exists and is not just empty text.
    articles_df["title"].notna() & (articles_df["title"].astype(str).str.strip() != "")
].copy()

print(f"Rows after dropping empty titles: {len(articles_df)}")            # Print the remaining row count after removing articles with empty titles.

if "url" in articles_df.columns:                                          # Check whether the dataframe includes a URL column for stronger deduplication.
    articles_df["url"] = articles_df["url"].fillna("").astype(str).str.strip()  # Clean URL values and replace missing ones with empty strings.

    with_url = articles_df[articles_df["url"] != ""].copy()               # Keep rows that have a usable URL.
    without_url = articles_df[articles_df["url"] == ""].copy()            # Keep rows that do not have a usable URL.

    before_url_dedup = len(with_url)                                      # Save the row count before deduplicating URL-based articles.
    with_url = with_url.drop_duplicates(subset=["url"]).copy()            # Remove duplicate rows that share the same URL.
    after_url_dedup = len(with_url)                                       # Save the row count after URL deduplication.

    articles_df = pd.concat([with_url, without_url], ignore_index=True)   # Combine the deduplicated URL rows back together with rows that had no URL.

    print(f"Rows with URL before dedup: {before_url_dedup}")              # Print how many rows had URLs before deduplication.
    print(f"Rows with URL after dedup : {after_url_dedup}")               # Print how many URL rows remained after deduplication.
    print(f"Total rows after URL dedup: {len(articles_df)}")              # Print total rows after recombining both groups.

articles_df["title"] = articles_df["title"].astype(str).str.strip()       # Clean the title column again so title-based deduplication works consistently.
articles_df = articles_df.drop_duplicates(subset=["title", "url"]).copy() # Apply a second lighter deduplication using title + URL together.

print(f"Rows after final dedup: {len(articles_df)}")                      # Print the final row count after all deduplication steps.

if len(articles_df) > MAX_ARTICLES:                                       # Check whether the dataframe is larger than the maximum allowed size for one run.
    articles_df = articles_df.head(MAX_ARTICLES).copy()                   # Keep only the first MAX_ARTICLES rows to control runtime and memory usage.
    print(f"Trimmed to {MAX_ARTICLES} articles for embedding generation") # Print a message showing that the dataset was trimmed.

articles_df["article_id"] = articles_df.apply(make_article_id, axis=1)    # Create a stable article_id for every final article row.
articles_df["embedding_text"] = articles_df.apply(build_text, axis=1)     # Build the text string that will be passed into the embedding model.

articles_df = articles_df[                                                # Keep only rows where the embedding text exists and is not empty.
    articles_df["embedding_text"].notna()
    & (articles_df["embedding_text"].astype(str).str.strip() != "")
].copy()

print(f"Final articles to embed: {len(articles_df)}")                     # Print the final number of articles that will actually be embedded.

texts = articles_df["embedding_text"].tolist()                            # Convert the embedding_text column into a normal Python list for model input.

print("Generating embeddings...")                                         # Print a message before the model starts embedding generation.
embeddings = model.encode(                                                # Generate embedding vectors from the article text list.
    texts,                                                                # Use the prepared text list as model input.
    batch_size=64,                                                        # Process texts in batches of 64 for better performance.
    show_progress_bar=True,                                               # Show a progress bar in the terminal while the embeddings are being generated.
    convert_to_numpy=True,                                                # Return the embeddings as a NumPy array.
)
print(f"Embeddings generated with shape: {embeddings.shape}")             # Print the final embedding matrix shape to verify the output dimensions.

print(f"Writing items to DynamoDB table: {DYNAMO_TABLE}")                 # Print a message to know the script is starting the DynamoDB write step.

success_count = 0                                                         # Create a counter to track how many items were written successfully.
error_count = 0                                                           # Create a counter to track how many rows failed during DynamoDB writing.

with table.batch_writer(overwrite_by_pkeys=["article_id"]) as batch:      # Open a DynamoDB batch writer so items can be written more efficiently in groups.
    for (_, row), vector in zip(articles_df.iterrows(), embeddings):      # Loop through each article row together with its matching embedding vector.
        try:                                                              # Start a try block so one bad row does not crash the entire script.
            embedding_list = [to_decimal(v) for v in vector.tolist()]     # Convert the numeric embedding vector into a DynamoDB-safe Decimal list.

            item = {                                                      # Build one DynamoDB item containing article metadata and the embedding.
                "article_id": row["article_id"],                          # Save the stable article_id as the DynamoDB partition key.
                "title": str(row.get("title", "")),                       # Save the article title.
                "url": str(row.get("url", "")),                           # Save the article URL.
                "category": str(row.get("category", "")),                 # Save the article category.
                "source": str(row.get("source_name", row.get("source", ""))),  # Save the article source using source_name first if it exists.
                "abstract": str(row.get("abstract", row.get("description", ""))),  # Save abstract or description as the text summary field.
                "pub_date": str(row.get("publishedAt", row.get("pub_date", ""))),  # Save publishedAt or pub_date as publication date text.
                "fetched_at": str(row.get("fetched_at", "")),             # Save the fetch timestamp if available.
                "data_source": str(row.get("data_source", "newsapi")),    # Save the source label such as newsapi or mind.
                "embedding": embedding_list,                              # Save the embedding vector itself as a list of Decimal values.
                "embedding_dim": int(len(embedding_list)),                # Save the embedding dimension size for quick checking later.
            }

            batch.put_item(Item=item)                                     # Write the item into DynamoDB using the batch writer.
            success_count += 1                                            # Increase the success counter after a successful write.

        except Exception as e:                                            # Catch any row-level error during DynamoDB writing.
            error_count += 1                                              # Increase the error counter when a row fails.
            print(f"Error writing row to DynamoDB: {e}")                  # Print the error but continue processing the rest of the rows.

print("Done.")                                                            # Print a final message when the whole script finishes.
print(f"Successfully wrote: {success_count}")                             # Print the total number of successfully written DynamoDB items.
print(f"Errors: {error_count}")                                           # Print the total number of row-level errors.
