# download_training_data.py

import os
import tempfile

import boto3
import pandas as pd


# ------------------------------
# Config
# ------------------------------
PROC_BUCKET = "news-recommending-processed-st125934"
RAW_BUCKET = "news-recommending-raw-st125934"
AWS_REGION = "ap-southeast-7"

# local output folder
LOCAL_OUTPUT_DIR = "training_data"


# ------------------------------
# AWS client
# ------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)


# ------------------------------
# Helper: list parquet files
# ------------------------------
def list_parquet_files(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    parquet_keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                parquet_keys.append(key)

    return parquet_keys


# ------------------------------
# Helper: download parquet folder
# ------------------------------
def download_parquet_folder(bucket, prefix, label):
    files = list_parquet_files(bucket, prefix)

    if not files:
        print(f"No parquet files found at s3://{bucket}/{prefix}")
        return pd.DataFrame()

    tmp_dir = tempfile.gettempdir()
    dfs = []

    for key in files:
        local_path = os.path.join(tmp_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        dfs.append(pd.read_parquet(local_path))

    df = pd.concat(dfs, ignore_index=True)
    print(f"{label}: {len(df)} rows loaded")
    return df


# ------------------------------
# Main
# ------------------------------
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

print("Downloading processed datasets from S3...")

# processed layer from Glue ETL
articles_df = download_parquet_folder(
    PROC_BUCKET,
    "articles/",
    "Articles"
)

behaviors_df = download_parquet_folder(
    PROC_BUCKET,
    "users/behaviors/",
    "Behaviors"
)

vectors_df = download_parquet_folder(
    PROC_BUCKET,
    "users/vectors/",
    "User vectors"
)

logs_df = download_parquet_folder(
    PROC_BUCKET,
    "users/logs/",
    "Processed user logs"
)

# save local copies for Phase 5 step 2
if not articles_df.empty:
    articles_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "articles.parquet"), index=False)

if not behaviors_df.empty:
    behaviors_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "behaviors.parquet"), index=False)

if not vectors_df.empty:
    vectors_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "user_vectors.parquet"), index=False)

if not logs_df.empty:
    logs_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "user_logs.parquet"), index=False)

print("\nSaved local files:")
for filename in os.listdir(LOCAL_OUTPUT_DIR):
    print(f"- {filename}")

print("\nPhase 5 Step 1 complete.")