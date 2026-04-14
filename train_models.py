import os                                                                                         # Import os so the script can build file paths safely on computer.
import pickle                                                                                     # Import pickle so trained model objects can be saved as .pkl files for the next phase.
import numpy as np                                                                                # Import numpy for numeric operations used in scoring and ranking.
import pandas as pd                                                                               # Import pandas for loading parquet files and preparing training data.
import mlflow                                                                                     # Import mlflow so training runs, params, and metrics can be logged locally.
from scipy.sparse import csr_matrix                                                               # Import csr_matrix so collaborative interactions can be stored efficiently as a sparse matrix.
from sklearn.decomposition import TruncatedSVD                                                    # Import TruncatedSVD to replace scikit-surprise SVD with pure scikit-learn matrix factorisation.
from sklearn.feature_extraction.text import TfidfVectorizer                                       # Import TfidfVectorizer to build article text features for the KNN content-based model.
from sklearn.neighbors import NearestNeighbors                                                    # Import NearestNeighbors to retrieve similar articles for both SVD-latent and content-based models.


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                                             # Get the folder where this current script is stored.
DATA_DIR = os.path.join(BASE_DIR, "training_data")                                                # Point to the local folder created by the previous Phase 5 steps.

TRAIN_PATH = os.path.join(DATA_DIR, "train_interactions.parquet")                                 # Local parquet file containing the training interactions produced by prepare_training_data.py.
TEST_PATH = os.path.join(DATA_DIR, "test_interactions.parquet")                                   # Local parquet file containing the testing interactions produced by prepare_training_data.py.
ARTICLES_PATH = os.path.join(DATA_DIR, "training_articles.parquet")                               # Local parquet file containing article metadata aligned to the final training interactions.

MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_DIR, 'mlruns')}"                                  # Use a local MLflow tracking directory inside the training folder.
MLFLOW_EXPERIMENT = "news-recommender"                                                            # Experiment name used later by save_best_model.py to find the best run.

RANDOM_STATE = 42                                                                                 # Fixed random seed so training behavior stays reproducible every run.
TOP_K_PRECISION = 5                                                                               # k value used for Precision@5 evaluation.
TOP_K_RECALL = 10                                                                                 # k value used for Recall@10 evaluation.
TOP_K_NDCG = 10                                                                                   # k value used for nDCG@10 evaluation.
MAX_EVAL_USERS = 5000                                                                             # Cap the number of evaluated users so the script stays fast on a student laptop.
SVD_COMPONENTS = 50                                                                               # Target latent factor count for TruncatedSVD before automatic shape adjustment.
NEIGHBOR_CANDIDATES = 50                                                                          # Number of nearest-neighbor candidates retrieved before filtering out seen items.
MAX_SEED_ITEMS = 3                                                                                # Number of user history items used as seeds when generating recommendations.


def precision_at_k(recommended, relevant_set, k):                                                 # Define a helper function to compute Precision@k for one user.
    if k == 0:                                                                                    # Guard against invalid k values.
        return 0.0                                                                                # Return zero if k is zero.
    hits = sum(1 for item in recommended[:k] if item in relevant_set)                            # Count how many recommended items in the top-k are relevant.
    return hits / float(k)                                                                        # Divide hits by k to get precision.


def recall_at_k(recommended, relevant_set, k):                                                    # Define a helper function to compute Recall@k for one user.
    if not relevant_set:                                                                          # Check whether the user has no relevant test items.
        return 0.0                                                                                # Return zero recall when there is no relevant ground truth.
    hits = sum(1 for item in recommended[:k] if item in relevant_set)                            # Count how many relevant items were recovered in the top-k list.
    return hits / float(len(relevant_set))                                                        # Divide hits by the number of relevant items to get recall.


def ndcg_at_k(recommended, relevant_set, k):                                                      # Define a helper function to compute nDCG@k for one user.
    if not relevant_set:                                                                          # Check whether the user has no relevant test items.
        return 0.0                                                                                # Return zero nDCG when there is no relevant ground truth.
    dcg = 0.0                                                                                     # Start discounted cumulative gain at zero.
    for rank, item in enumerate(recommended[:k], start=1):                                        # Loop through the top-k recommendations with 1-based ranking positions.
        if item in relevant_set:                                                                  # Check whether the current recommended item is relevant.
            dcg += 1.0 / np.log2(rank + 1)                                                        # Add discounted gain for this relevant item.
    ideal_hits = min(len(relevant_set), k)                                                        # The ideal list can contain at most min(number of relevant items, k) relevant hits.
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))                     # Compute ideal discounted cumulative gain for normalization.
    if idcg == 0:                                                                                 # Guard against division by zero.
        return 0.0                                                                                # Return zero if ideal DCG is zero.
    return dcg / idcg                                                                             # Divide DCG by IDCG to get normalized DCG.


def fill_with_popular(base_recs, seen_set, popular_articles, k):                                 # Define a helper function that tops up a recommendation list with popular fallback items.
    final_recs = []                                                                               # Create an empty list for the final recommendations.
    used = set()                                                                                  # Track which items have already been added so duplicates are avoided.

    for article_id in base_recs:                                                                  # Loop through the first-stage recommendation candidates.
        if article_id not in seen_set and article_id not in used:                                 # Keep only unseen and not-yet-added items.
            final_recs.append(article_id)                                                         # Add the candidate to the final list.
            used.add(article_id)                                                                  # Mark this candidate as already used.
        if len(final_recs) >= k:                                                                  # Stop early once the target length is reached.
            return final_recs[:k]                                                                 # Return the top-k final recommendations.

    for article_id in popular_articles:                                                           # If the first-stage list was too short, fill from the popularity fallback list.
        if article_id not in seen_set and article_id not in used:                                 # Again keep only unseen and unique items.
            final_recs.append(article_id)                                                         # Add the popular fallback item.
            used.add(article_id)                                                                  # Mark it as used.
        if len(final_recs) >= k:                                                                  # Stop once top-k length is reached.
            break                                                                                 # Break the fallback loop.

    return final_recs[:k]                                                                         # Return the final recommendation list limited to k items.


def evaluate_model(model_name, recommender_fn, eval_user_ids, train_histories, test_truth):      # Define a helper function that evaluates one model over the chosen evaluation users.
    precision_scores = []                                                                         # Store per-user Precision@5 scores here.
    recall_scores = []                                                                            # Store per-user Recall@10 scores here.
    ndcg_scores = []                                                                              # Store per-user nDCG@10 scores here.

    for user_id in eval_user_ids:                                                                 # Loop through each evaluation user.
        seen_set = set(train_histories.get(user_id, []))                                          # Build the set of items the user already clicked in training.
        relevant_set = set(test_truth.get(user_id, []))                                           # Build the set of held-out relevant items for testing.
        if not relevant_set:                                                                      # Skip users with no test ground truth.
            continue                                                                              # Continue to the next user.

        recs = recommender_fn(user_id, seen_set, TOP_K_NDCG)                                      # Ask the model to generate recommendations up to top 10.
        precision_scores.append(precision_at_k(recs, relevant_set, TOP_K_PRECISION))              # Compute Precision@5 for this user and store it.
        recall_scores.append(recall_at_k(recs, relevant_set, TOP_K_RECALL))                       # Compute Recall@10 for this user and store it.
        ndcg_scores.append(ndcg_at_k(recs, relevant_set, TOP_K_NDCG))                             # Compute nDCG@10 for this user and store it.

    metrics = {                                                                                   # Build one clean metrics dictionary for this model.
        "precision_at_5": float(np.mean(precision_scores)) if precision_scores else 0.0,          # Average Precision@5 across evaluated users.
        "recall_at_10": float(np.mean(recall_scores)) if recall_scores else 0.0,                  # Average Recall@10 across evaluated users.
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,                        # Average nDCG@10 across evaluated users.
    }

    print(f"{model_name} metrics:")                                                               # Print a label before the metric values.
    print(f"  precision_at_5: {metrics['precision_at_5']:.4f}")                                   # Print Precision@5 in a readable format.
    print(f"  recall_at_10 : {metrics['recall_at_10']:.4f}")                                      # Print Recall@10 in a readable format.
    print(f"  ndcg_at_10   : {metrics['ndcg_at_10']:.4f}")                                        # Print nDCG@10 in a readable format.

    return metrics                                                                                # Return the final metrics dictionary.


print("Loading training data...")                                                                 # Print a start message for the terminal.
train_df = pd.read_parquet(TRAIN_PATH)                                                            # Load the train interactions parquet produced by prepare_training_data.py.
test_df = pd.read_parquet(TEST_PATH)                                                              # Load the test interactions parquet produced by prepare_training_data.py.
articles_df = pd.read_parquet(ARTICLES_PATH)                                                      # Load the filtered article metadata aligned to the training interactions.

train_df["user_id"] = train_df["user_id"].astype(str)                                             # Force train user IDs to string for consistency.
train_df["article_id"] = train_df["article_id"].astype(str)                                       # Force train article IDs to string for consistency.
test_df["user_id"] = test_df["user_id"].astype(str)                                               # Force test user IDs to string for consistency.
test_df["article_id"] = test_df["article_id"].astype(str)                                         # Force test article IDs to string for consistency.
articles_df["article_id"] = articles_df["article_id"].astype(str)                                 # Force article metadata IDs to string for consistency.

print(f"Train: {len(train_df)} | Test: {len(test_df)} | Users: {train_df['user_id'].nunique()} | Articles: {train_df['article_id'].nunique()}")  # Print the main dataset sizes for quick verification.

train_histories = train_df.groupby("user_id")["article_id"].apply(list).to_dict()                 # Build a dictionary of each user's clicked articles from the training set.
test_truth = test_df.groupby("user_id")["article_id"].apply(list).to_dict()                        # Build a dictionary of each user's held-out clicked articles from the test set.

eval_user_ids = list(test_truth.keys())                                                            # Start with all users who appear in the test set.
if len(eval_user_ids) > MAX_EVAL_USERS:                                                            # Check whether the test user set is larger than the runtime cap.
    rng = np.random.default_rng(RANDOM_STATE)                                                      # Create a deterministic random generator using the fixed seed.
    eval_user_ids = rng.choice(eval_user_ids, size=MAX_EVAL_USERS, replace=False).tolist()        # Sample a fixed-size subset of test users for faster evaluation.

print(f"Evaluation users: {len(eval_user_ids)}")                                                   # Print how many users will actually be evaluated.

popular_articles = train_df["article_id"].value_counts().index.astype(str).tolist()                # Build a global popularity ranking of articles from most clicked to least clicked.


def recommend_popularity(user_id, seen_set, k):                                                    # Define the popularity-baseline recommendation function.
    return fill_with_popular([], seen_set, popular_articles, k)                                    # Return the top-k popular articles while excluding anything already seen by the user.


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)                                                       # Tell MLflow to store experiment data in the local mlruns folder.
mlflow.set_experiment(MLFLOW_EXPERIMENT)                                                           # Create or select the local MLflow experiment named news-recommender.
mlflow.end_run()                                                                                   # Make sure no old MLflow run is still active before starting a new one.


print("Training Model 1: Popularity Baseline...")                                                  # Print a message to know that the first model is starting.
popularity_model = {                                                                               # Create a simple model object for the popularity baseline.
    "model_type": "popularity_baseline",                                                           # Save the model type name used later by save_best_model.py.
    "popular_articles": popular_articles,                                                          # Save the global popularity ranking for later serving.
}                                                                                                  # Finish the popularity model dictionary.

with mlflow.start_run(run_name="popularity_baseline"):                                             # Start an MLflow run for the popularity baseline model.
    with open("model_popularity.pkl", "wb") as f:                                                  # Open the local popularity model file for binary writing.
        pickle.dump(popularity_model, f)                                                           # Save the popularity model object as a pickle file.

    popularity_metrics = evaluate_model(                                                           # Evaluate the popularity recommender on the evaluation users.
        "Popularity baseline",                                                                     # Human-readable model name for terminal output.
        recommend_popularity,                                                                      # Recommendation function used during evaluation.
        eval_user_ids,                                                                             # Evaluation user list.
        train_histories,                                                                           # Training histories for seen-item exclusion.
        test_truth,                                                                                # Test ground truth for metrics.
    )
    mlflow.log_param("model_type", "popularity_baseline")                                          # Log the popularity model type as an MLflow parameter.
    mlflow.log_param("train_size", len(train_df))                                                  # Log the train interaction count as a parameter.
    mlflow.log_param("test_size", len(test_df))                                                    # Log the test interaction count as a parameter.
    mlflow.log_param("unique_users", train_df["user_id"].nunique())                                # Log the number of unique training users.
    mlflow.log_param("unique_articles", train_df["article_id"].nunique())                          # Log the number of unique training articles.
    mlflow.log_param("eval_users", len(eval_user_ids))                                             # Log the number of users actually evaluated.
    mlflow.log_metrics(popularity_metrics)                                                         # Log all popularity baseline metrics into MLflow.
    mlflow.log_artifact("model_popularity.pkl")                                                    # Log the popularity model pickle as an MLflow artifact inside the correct run.

mlflow.end_run()                                                                                   # Explicitly close any active run before starting the next model.


print("Training Model 2: SVD Matrix Factorisation...")                                             # Print a message to know the second model is starting.
svd_counts = train_df.groupby(["user_id", "article_id"]).size().reset_index(name="weight")        # Aggregate duplicate user-article interactions into weighted counts.
svd_user_ids = svd_counts["user_id"].drop_duplicates().tolist()                                    # Build the ordered list of training users used in the matrix.
svd_article_ids = svd_counts["article_id"].drop_duplicates().tolist()                              # Build the ordered list of training articles used in the matrix.

svd_user_index = {user_id: idx for idx, user_id in enumerate(svd_user_ids)}                        # Map each user_id to its sparse-matrix row index.
svd_article_index = {article_id: idx for idx, article_id in enumerate(svd_article_ids)}            # Map each article_id to its sparse-matrix column index.

row_idx = svd_counts["user_id"].map(svd_user_index).to_numpy()                                     # Convert user IDs into row positions for the sparse matrix.
col_idx = svd_counts["article_id"].map(svd_article_index).to_numpy()                               # Convert article IDs into column positions for the sparse matrix.
weights = svd_counts["weight"].astype(float).to_numpy()                                            # Convert aggregated interaction counts into numeric weights.

interaction_matrix = csr_matrix(                                                                   # Build a sparse user-article interaction matrix.
    (weights, (row_idx, col_idx)),                                                                 # Use weights as values with row and column index arrays.
    shape=(len(svd_user_ids), len(svd_article_ids)),                                               # Set the matrix shape as users x articles.
)

item_user_matrix = interaction_matrix.T                                                            # Transpose the interaction matrix so items become rows for item-latent factor learning.
max_components = min(item_user_matrix.shape) - 1                                                   # Compute the maximum safe latent dimension for TruncatedSVD.
n_components = max(2, min(SVD_COMPONENTS, max_components))                                         # Choose a safe latent dimension count with a minimum of 2.

svd_model = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)                     # Create the TruncatedSVD model using the chosen latent dimension count.
svd_item_factors = svd_model.fit_transform(item_user_matrix)                                       # Learn latent factor vectors for each training article.
svd_knn = NearestNeighbors(metric="cosine", algorithm="brute")                                     # Create a nearest-neighbor model using cosine distance in latent space.
svd_knn.fit(svd_item_factors)                                                                      # Fit the nearest-neighbor model on the learned SVD item factors.


def recommend_svd(user_id, seen_set, k):                                                           # Define the SVD recommendation function using nearest neighbors in latent item space.
    history = train_histories.get(user_id, [])                                                     # Fetch the user's clicked training history.
    if not history:                                                                                # Check whether the user has no training history at all.
        return fill_with_popular([], seen_set, popular_articles, k)                                # Fall back to popularity if the user has no history.

    seed_items = [item for item in history[-MAX_SEED_ITEMS:] if item in svd_article_index]        # Keep the last few clicked items that actually exist in the SVD article index.
    if not seed_items:                                                                             # Check whether none of the user's seeds are usable.
        return fill_with_popular([], seen_set, popular_articles, k)                                # Fall back to popularity if no seed item is usable.

    scores = {}                                                                                    # Create a score dictionary to accumulate candidate recommendation strength.
    n_neighbors = min(NEIGHBOR_CANDIDATES, len(svd_article_ids))                                   # Choose a safe neighbor count based on the article universe size.

    for seed_item in seed_items:                                                                   # Loop through each usable seed item.
        seed_idx = svd_article_index[seed_item]                                                    # Convert the seed article ID into its latent-factor row index.
        distances, indices = svd_knn.kneighbors(svd_item_factors[[seed_idx]], n_neighbors=n_neighbors)  # Retrieve the nearest latent-space neighbors for the seed item.

        for dist, idx in zip(distances[0], indices[0]):                                            # Loop through each returned neighbor candidate.
            candidate = svd_article_ids[idx]                                                       # Convert the neighbor index back into article_id.
            if candidate == seed_item or candidate in seen_set:                                     # Skip the seed itself and any article the user has already seen.
                continue                                                                            # Continue to the next neighbor candidate.
            scores[candidate] = scores.get(candidate, 0.0) + (1.0 - float(dist))                  # Accumulate similarity-based score using cosine similarity proxy.

    ranked = [article_id for article_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]  # Sort candidates by descending accumulated score.
    return fill_with_popular(ranked, seen_set, popular_articles, k)                                # Fill any missing slots with popular fallback articles.


with mlflow.start_run(run_name="svd_matrix_factorisation"):                                        # Start an MLflow run for the SVD matrix factorisation model.
    svd_metrics = evaluate_model(                                                                  # Evaluate the SVD recommender on the evaluation users.
        "SVD",                                                                                     # Human-readable model name for terminal output.
        recommend_svd,                                                                             # Recommendation function used during evaluation.
        eval_user_ids,                                                                             # Evaluation user list.
        train_histories,                                                                           # Training histories for seen-item exclusion.
        test_truth,                                                                                # Test ground truth for metrics.
    )

    svd_pickle = {                                                                                 # Build a serializable model object for SVD serving later.
        "model_type": "svd_matrix_factorisation",                                                  # Save the updated model type name expected by the later registry step.
        "article_ids": svd_article_ids,                                                            # Save the ordered article ID list used by the latent factors.
        "article_index": svd_article_index,                                                        # Save the article_id -> row index mapping.
        "item_factors": svd_item_factors,                                                          # Save the learned latent item factor matrix.
        "knn_model": svd_knn,                                                                      # Save the nearest-neighbor model built on the latent item space.
        "popular_articles": popular_articles,                                                      # Save the popularity fallback list for cold-start handling.
        "max_seed_items": MAX_SEED_ITEMS,                                                          # Save the number of seed history items used during recommendation.
    }

    with open("model_svd.pkl", "wb") as f:                                                         # Open the local SVD model file for binary writing.
        pickle.dump(svd_pickle, f)                                                                 # Save the SVD model object as a pickle file.

    mlflow.log_param("model_type", "svd_matrix_factorisation")                                     # Log the updated SVD model type expected by save_best_model.py.
    mlflow.log_param("train_size", len(train_df))                                                  # Log the train interaction count as a parameter.
    mlflow.log_param("test_size", len(test_df))                                                    # Log the test interaction count as a parameter.
    mlflow.log_param("unique_users", train_df["user_id"].nunique())                                # Log the number of unique training users.
    mlflow.log_param("unique_articles", train_df["article_id"].nunique())                          # Log the number of unique training articles.
    mlflow.log_param("eval_users", len(eval_user_ids))                                             # Log the number of users actually evaluated.
    mlflow.log_param("svd_components", n_components)                                               # Log the latent dimension count used by TruncatedSVD.
    mlflow.log_param("neighbor_candidates", NEIGHBOR_CANDIDATES)                                   # Log the number of nearest neighbors searched per seed item.
    mlflow.log_metrics(svd_metrics)                                                                # Log all SVD metrics into MLflow.
    mlflow.log_artifact("model_svd.pkl")                                                           # Log the SVD model pickle as an MLflow artifact inside the correct run.

mlflow.end_run()                                                                                   # Explicitly close any active run before starting the next model.


print("Training Model 3: KNN Content-Based...")                                                    # Print a message to know the third model is starting.
articles_df["title"] = articles_df["title"].fillna("").astype(str)                                 # Clean the title column so text building is safe.
if "abstract" in articles_df.columns:                                                              # Check whether an abstract column exists in the article metadata.
    articles_df["abstract"] = articles_df["abstract"].fillna("").astype(str)                       # Clean the abstract column if it exists.
else:                                                                                              # Handle the case where abstract is missing.
    articles_df["abstract"] = ""                                                                   # Create an empty abstract column so the next text-building step still works.

articles_df["text"] = (articles_df["title"] + " " + articles_df["abstract"]).str.strip()          # Build one combined text field from title and abstract.
articles_df = articles_df[articles_df["text"] != ""].drop_duplicates(subset=["article_id"]).copy()# Keep only articles with usable text and unique article IDs.

knn_article_ids = articles_df["article_id"].astype(str).tolist()                                   # Build the ordered article ID list used by the content model.
knn_article_index = {article_id: idx for idx, article_id in enumerate(knn_article_ids)}            # Map each content-model article_id to its row index.

tfidf_vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")                        # Create a TF-IDF vectorizer for article text content.
tfidf_matrix = tfidf_vectorizer.fit_transform(articles_df["text"])                                  # Convert article text into a sparse TF-IDF matrix.
content_knn = NearestNeighbors(metric="cosine", algorithm="brute")                                  # Create a nearest-neighbor model using cosine distance on TF-IDF vectors.
content_knn.fit(tfidf_matrix)                                                                       # Fit the content nearest-neighbor model on the article TF-IDF matrix.


def recommend_knn(user_id, seen_set, k):                                                            # Define the content-based KNN recommendation function.
    history = train_histories.get(user_id, [])                                                      # Fetch the user's clicked training history.
    if not history:                                                                                 # Check whether the user has no training history.
        return fill_with_popular([], seen_set, popular_articles, k)                                 # Fall back to popularity if the user has no history.

    seed_items = [item for item in history[-MAX_SEED_ITEMS:] if item in knn_article_index]         # Keep only the last few clicked items that exist in the content article index.
    if not seed_items:                                                                              # Check whether none of the user's seeds are usable.
        return fill_with_popular([], seen_set, popular_articles, k)                                 # Fall back to popularity if no seed item is usable.

    scores = {}                                                                                     # Create a score dictionary to accumulate candidate recommendation strength.
    n_neighbors = min(NEIGHBOR_CANDIDATES, len(knn_article_ids))                                    # Choose a safe neighbor count based on the article universe size.

    for seed_item in seed_items:                                                                    # Loop through each usable seed item.
        seed_idx = knn_article_index[seed_item]                                                     # Convert the seed article ID into its TF-IDF row index.
        distances, indices = content_knn.kneighbors(tfidf_matrix[seed_idx], n_neighbors=n_neighbors)  # Retrieve the nearest text neighbors for the seed item.

        for dist, idx in zip(distances[0], indices[0]):                                             # Loop through each returned neighbor candidate.
            candidate = knn_article_ids[idx]                                                        # Convert the neighbor index back into article_id.
            if candidate == seed_item or candidate in seen_set:                                      # Skip the seed itself and any article the user has already seen.
                continue                                                                             # Continue to the next neighbor candidate.
            scores[candidate] = scores.get(candidate, 0.0) + (1.0 - float(dist))                   # Accumulate similarity-based score using cosine similarity proxy.

    ranked = [article_id for article_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]  # Sort candidates by descending accumulated score.
    return fill_with_popular(ranked, seen_set, popular_articles, k)                                 # Fill any missing slots with popular fallback articles.


with mlflow.start_run(run_name="knn_content_based"):                                                # Start an MLflow run for the KNN content-based model.
    knn_metrics = evaluate_model(                                                                   # Evaluate the KNN recommender on the evaluation users.
        "KNN",                                                                                      # Human-readable model name for terminal output.
        recommend_knn,                                                                              # Recommendation function used during evaluation.
        eval_user_ids,                                                                              # Evaluation user list.
        train_histories,                                                                            # Training histories for seen-item exclusion.
        test_truth,                                                                                 # Test ground truth for metrics.
    )

    knn_pickle = {                                                                                  # Build a serializable model object for content-based KNN serving later.
        "model_type": "knn_content_based",                                                          # Save the KNN model type expected by save_best_model.py.
        "article_ids": knn_article_ids,                                                             # Save the ordered article ID list used by the content model.
        "article_index": knn_article_index,                                                         # Save the article_id -> row index mapping.
        "vectorizer": tfidf_vectorizer,                                                             # Save the fitted TF-IDF vectorizer used to encode article text.
        "knn_model": content_knn,                                                                   # Save the nearest-neighbor model built on article TF-IDF vectors.
        "tfidf_matrix": tfidf_matrix,                                                               # Save the TF-IDF matrix so recommendation can query nearest neighbors later.
        "popular_articles": popular_articles,                                                       # Save the popularity fallback list for cold-start handling.
        "max_seed_items": MAX_SEED_ITEMS,                                                           # Save the number of seed history items used during recommendation.
    }

    with open("model_knn.pkl", "wb") as f:                                                          # Open the local KNN model file for binary writing.
        pickle.dump(knn_pickle, f)                                                                  # Save the KNN model object as a pickle file.

    mlflow.log_param("model_type", "knn_content_based")                                             # Log the KNN model type expected by save_best_model.py.
    mlflow.log_param("train_size", len(train_df))                                                   # Log the train interaction count as a parameter.
    mlflow.log_param("test_size", len(test_df))                                                     # Log the test interaction count as a parameter.
    mlflow.log_param("unique_users", train_df["user_id"].nunique())                                 # Log the number of unique training users.
    mlflow.log_param("unique_articles", train_df["article_id"].nunique())                           # Log the number of unique training articles.
    mlflow.log_param("eval_users", len(eval_user_ids))                                              # Log the number of users actually evaluated.
    mlflow.log_param("neighbor_candidates", NEIGHBOR_CANDIDATES)                                    # Log the number of nearest neighbors searched per seed item.
    mlflow.log_metrics(knn_metrics)                                                                 # Log all KNN metrics into MLflow.
    mlflow.log_artifact("model_knn.pkl")                                                            # Log the KNN model pickle as an MLflow artifact inside the correct run.

mlflow.end_run()                                                                                   # Explicitly close any active run after the last model finishes.


print("All models trained successfully.")                                                           # Print the final success message for the full training step.