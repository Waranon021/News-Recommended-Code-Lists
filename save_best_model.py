import os                                                                 # Import os so the script can build file paths safely on computer.
import json                                                               # Import json so the script can save model metadata as a JSON file.
import hashlib                                                            # Import hashlib so the script can generate a data version hash for this model run.
from datetime import datetime, timezone                                   # Import datetime tools so uploaded model files can have a UTC timestamp.
from pathlib import Path                                                  # Import Path so local file handling is easier and clearer.

import boto3                                                              # Import boto3 so the script can upload model files and metadata to S3.
import mlflow                                                             # Import mlflow so the script can read experiment runs and metrics.
from mlflow.tracking import MlflowClient                                  # Import MlflowClient so the script can query the local MLflow experiment cleanly.


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # Get the folder where this current script is stored.
MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_DIR, 'mlruns')}"          # Point MLflow to the local mlruns folder created by train_models.py.
MLFLOW_EXPERIMENT = "news-recommender"                                    # Name of the MLflow experiment used in train_models.py.

MODEL_BUCKET = "news-recommending-models-st125934"                        # Name of the S3 bucket where trained model files will be stored.
AWS_REGION = "ap-southeast-7"                                             # AWS region for your current account, which is Thailand.

MODEL_FILE_MAP = {                                                        # Map each model_type from train_models.py to its local pickle file name.
    "popularity_baseline": "model_popularity.pkl",                        # Popularity model local pickle file.
    "svd_matrix_factorisation": "model_svd.pkl",                          # SVD model local pickle file.
    "knn_content_based": "model_knn.pkl",                                 # KNN content-based model local pickle file.
}


s3 = boto3.client("s3", region_name=AWS_REGION)                           # Create one S3 client that will upload model files and metadata to the models bucket.


def safe_float(value, default=0.0):                                       # Define a helper function that safely converts MLflow metric values into float.
    try:                                                                  # Start a try block so bad or missing values do not crash the script.
        return float(value)                                               # Return the metric as float when conversion works.
    except Exception:                                                     # Catch any error from missing or malformed values.
        return float(default)                                             # Return the fallback default value if conversion fails.


print("Finding best model from MLflow...")                                # Print a start message so you know the script is beginning the registry step.

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)                              # Tell MLflow to read runs from the local mlruns folder.
client = MlflowClient()                                                   # Create an MLflow client used to search the experiment runs.

experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)             # Find the MLflow experiment created in train_models.py by its name.
if experiment is None:                                                    # Check whether the experiment does not exist at all.
    raise ValueError(f"MLflow experiment not found: {MLFLOW_EXPERIMENT}") # Stop with a clear error if training was not run yet.

runs = client.search_runs(                                                # Search all runs inside the selected experiment.
    experiment_ids=[experiment.experiment_id],                            # Use the current experiment ID as the search target.
    filter_string="attributes.status = 'FINISHED'",                       # Keep only finished MLflow runs.
    max_results=100,                                                      # Limit the number of returned runs to a reasonable amount.
)                                                                         # Finish the MLflow run search.

candidate_runs = []                                                       # Create a list to store only the valid model-training runs.

for run in runs:                                                          # Loop through each MLflow run returned from the experiment.
    model_type = run.data.params.get("model_type", "")                    # Read the model_type param logged during train_models.py.
    if model_type in MODEL_FILE_MAP:                                      # Keep only runs whose model_type matches one of the three supported models.
        ndcg = safe_float(run.data.metrics.get("ndcg_at_10", 0.0))        # Read nDCG@10 from the MLflow metrics and convert it safely.
        precision = safe_float(run.data.metrics.get("precision_at_5", 0.0))  # Read Precision@5 from the MLflow metrics and convert it safely.
        recall = safe_float(run.data.metrics.get("recall_at_10", 0.0))    # Read Recall@10 from the MLflow metrics and convert it safely.
        candidate_runs.append((run, model_type, ndcg, precision, recall)) # Save the run together with its key metrics for ranking.

if not candidate_runs:                                                    # Check whether no valid model runs were found at all.
    raise ValueError("No valid trained model runs found in MLflow.")      # Stop with a clear error if Step 3 has not completed successfully.

best_run, best_model_name, best_ndcg, best_precision, best_recall = max(  # Pick the best run using nDCG first, then Precision, then Recall as tie-breakers.
    candidate_runs,                                                       # Use the list of valid candidate runs.
    key=lambda x: (x[2], x[3], x[4])                                      # Rank by ndcg_at_10, then precision_at_5, then recall_at_10.
)

print(f"Best model : {best_model_name}")                                  # Print the chosen best model name.
print(f"nDCG@10    : {best_ndcg:.4f}")                                    # Print the best model nDCG@10.
print(f"Precision@5: {best_precision:.4f}")                               # Print the best model Precision@5.
print(f"Recall@10  : {best_recall:.4f}")                                  # Print the best model Recall@10.

best_local_file = Path(BASE_DIR) / MODEL_FILE_MAP[best_model_name]        # Build the local file path of the winning model pickle.
if not best_local_file.exists():                                          # Check whether the winning local model file is missing.
    raise FileNotFoundError(f"Best model file not found locally: {best_local_file}")  # Stop with a clear error if the pickle file does not exist.

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")          # Create a UTC timestamp string for the registry file names.
model_key = f"registry/best_model_{timestamp}.pkl"                        # Build the timestamped S3 registry key for this best model version.
latest_key = "registry/latest_model.pkl"                                  # Build the fixed S3 key that serving code can always load.
metadata_key = "registry/model_metadata.json"                             # Build the S3 key for the metadata JSON file.

s3.upload_file(str(best_local_file), MODEL_BUCKET, model_key)             # Upload the winning model pickle to a timestamped registry path in S3.
print(f"Best model uploaded to s3://{MODEL_BUCKET}/{model_key}")          # Print the exact S3 path of the uploaded best model.

s3.upload_file(str(best_local_file), MODEL_BUCKET, latest_key)            # Upload the same winning model again as latest_model.pkl for serving.
print(f"Also saved as {latest_key}")                                      # Print confirmation that latest_model.pkl was updated.

for model_type, local_name in MODEL_FILE_MAP.items():                     # Loop through every locally saved model file so backup copies can be uploaded.
    local_path = Path(BASE_DIR) / local_name                              # Build the full local path for the current model file.
    if local_path.exists():                                               # Only upload the file if it really exists locally.
        backup_key = f"registry/all/{model_type}.pkl"                     # Build the S3 backup path for this specific model type.
        s3.upload_file(str(local_path), MODEL_BUCKET, backup_key)         # Upload the model pickle into the registry/all/ backup folder.
        print(f"Backup uploaded: {model_type}")                           # Print confirmation that this model backup was uploaded.

train_size = int(best_run.data.params.get("train_size", 0))               # Read the train interaction count from MLflow params instead of from train_df.
test_size = int(best_run.data.params.get("test_size", 0))                 # Read the test interaction count from MLflow params instead of from test_df.
unique_users = int(best_run.data.params.get("unique_users", 0))           # Read the unique training user count from MLflow params.
unique_articles = int(best_run.data.params.get("unique_articles", 0))     # Read the unique training article count from MLflow params.
eval_users = int(best_run.data.params.get("eval_users", 0))               # Read the evaluated user count from MLflow params.

data_version = hashlib.md5(                                               # Generate a short version hash so each retraining run gets a traceable data version.
    f"{best_run.info.run_id}_{timestamp}".encode()                        # Use MLflow run ID plus current timestamp to make the hash unique.
).hexdigest()[:8]                                                         # Keep only the first 8 characters so the version string stays short.

metadata = {                                                              # Build the final metadata dictionary for the registry.
    "best_model": best_model_name,                                        # Save the chosen best model type.
    "model_file": model_key,                                              # Save the timestamped S3 key of the best model.
    "timestamp": timestamp,                                               # Save the UTC timestamp of this registry update.
    "data_version": data_version,                                         # Save the generated data version hash.
    "training_stats": {                                                   # Save key training statistics pulled from MLflow params.
        "train_interactions": train_size,                                 # Number of train interactions used by train_models.py.
        "test_interactions": test_size,                                   # Number of test interactions used by train_models.py.
        "unique_users": unique_users,                                     # Number of unique users in the training data.
        "unique_articles": unique_articles,                               # Number of unique articles in the training data.
        "eval_users": eval_users,                                         # Number of users evaluated during model comparison.
        "mlflow_run_id": best_run.info.run_id,                            # MLflow run ID of the winning model.
        "experiment": MLFLOW_EXPERIMENT,                                  # Name of the MLflow experiment used.
    },
    "metrics": {                                                          # Save the winning model’s key evaluation metrics.
        "ndcg_at_10": best_ndcg,                                          # Best model nDCG@10.
        "precision_at_5": best_precision,                                 # Best model Precision@5.
        "recall_at_10": best_recall,                                      # Best model Recall@10.
    },
    "mlflow_run_id": best_run.info.run_id                                 # Save the winning MLflow run ID again at the top level for convenience.
}

metadata_path = Path(BASE_DIR) / "model_metadata.json"                    # Build the local path where the metadata JSON file will be saved temporarily.
with open(metadata_path, "w", encoding="utf-8") as f:                     # Open the local metadata JSON file for writing.
    json.dump(metadata, f, indent=2)                                      # Save the metadata dictionary as nicely formatted JSON.

s3.upload_file(str(metadata_path), MODEL_BUCKET, metadata_key)            # Upload the metadata JSON file into the S3 registry folder.
print("\nModel metadata saved.")                                          # Print confirmation that metadata was saved and uploaded.

print("\nPhase 5 complete. Best model is in S3 ready for serving.")       # Print the final success message for this whole phase step.