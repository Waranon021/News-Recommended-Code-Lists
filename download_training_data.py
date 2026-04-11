# download_training_data.py

import os                                                          # Import os so the script can build folder paths and save local files.
import tempfile                                                    # Import tempfile so downloaded parquet parts can be stored safely in a temporary folder first.

import boto3                                                       # Import boto3 so the script can connect to AWS S3 and download parquet files.
import pandas as pd                                                # Import pandas so the script can read parquet files and combine them into DataFrames.


PROC_BUCKET = "news-recommending-processed-st125934"               # Name of the processed S3 bucket that stores ETL outputs such as articles, behaviors, vectors, and logs.
RAW_BUCKET = "news-recommending-raw-st125934"                      # Name of the raw S3 bucket; not used directly here, but kept for consistency with the project config.
AWS_REGION = "ap-southeast-7"                                      # AWS region for your current account, which is Thailand (ap-southeast-7).

LOCAL_OUTPUT_DIR = "training_data"                                 # Local folder name where the downloaded parquet files will be saved for the next training step.


s3 = boto3.client("s3", region_name=AWS_REGION)                    # Create one S3 client object that all helper functions will use to talk to your bucket.


def list_parquet_files(bucket, prefix):                            # Define a helper function that lists every parquet file under a given S3 folder prefix.
    paginator = s3.get_paginator("list_objects_v2")                # Create a paginator because one S3 folder may contain many parquet files across multiple result pages.
    parquet_keys = []                                              # Create an empty list to store the matching parquet file keys.

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):  # Loop through every result page returned by S3 for the selected bucket and prefix.
        for obj in page.get("Contents", []):                       # Loop through every file object inside the current result page.
            key = obj["Key"]                                       # Extract the S3 key path of the current object.
            if key.endswith(".parquet"):                           # Keep only files that end with .parquet because those are the processed datasets we need.
                parquet_keys.append(key)                           # Add the parquet key into the list of files to download.

    return parquet_keys                                            # Return the full list of parquet file paths found in that S3 folder.


def download_parquet_folder(bucket, prefix, label):                # Define a helper function that downloads all parquet files from one S3 folder and combines them.
    files = list_parquet_files(bucket, prefix)                     # Call the previous helper function to get all parquet file keys under this prefix.

    if not files:                                                  # Check whether that S3 folder has no parquet files at all.
        print(f"No parquet files found at s3://{bucket}/{prefix}") # Print a message so you know that this specific dataset was missing.
        return pd.DataFrame()                                      # Return an empty DataFrame so the script can continue without crashing.

    tmp_dir = tempfile.gettempdir()                                # Get the system temporary folder path where parquet parts can be downloaded safely.
    dfs = []                                                       # Create an empty list to hold each parquet file after reading it into pandas.

    for key in files:                                              # Loop through each parquet file path found in that S3 folder.
        local_path = os.path.join(tmp_dir, os.path.basename(key))  # Build a temporary local path using only the parquet file name.
        s3.download_file(bucket, key, local_path)                  # Download the parquet file from S3 to the temporary local path.
        dfs.append(pd.read_parquet(local_path))                    # Read the downloaded parquet file into pandas and add it to the list.

    df = pd.concat(dfs, ignore_index=True)                         # Combine all parquet parts into one DataFrame and reset the row index.
    print(f"{label}: {len(df)} rows loaded")                       # Print a summary showing how many rows were loaded for this dataset.
    return df                                                      # Return the final combined DataFrame for this S3 folder.


os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)                       # Create the local training_data folder if it does not already exist.

print("Downloading processed datasets from S3...")                 # Print a start message so you know the script has begun Step 1.

articles_df = download_parquet_folder(                             # Download and combine the processed article parquet files from S3.
    PROC_BUCKET,                                                   # Use the processed bucket as the source.
    "articles/",                                                   # Read everything inside the articles/ folder.
    "Articles"                                                     # Use this label in the terminal print output.
)

behaviors_df = download_parquet_folder(                            # Download and combine the processed behaviors parquet files from S3.
    PROC_BUCKET,                                                   # Use the processed bucket as the source.
    "users/behaviors/",                                            # Read everything inside users/behaviors/.
    "Behaviors"                                                    # Use this label in the terminal print output.
)

vectors_df = download_parquet_folder(                              # Download and combine the processed user vectors parquet files from S3.
    PROC_BUCKET,                                                   # Use the processed bucket as the source.
    "users/vectors/",                                              # Read everything inside users/vectors/.
    "User vectors"                                                 # Use this label in the terminal print output.
)

logs_df = download_parquet_folder(                                 # Download and combine the processed user logs parquet files from S3.
    PROC_BUCKET,                                                   # Use the processed bucket as the source.
    "users/logs/",                                                 # Read everything inside users/logs/.
    "Processed user logs"                                          # Use this label in the terminal print output.
)

if not articles_df.empty:                                          # Check whether the articles dataset was downloaded successfully and is not empty.
    articles_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "articles.parquet"), index=False)  # Save the local combined articles file for the next training step.

if not behaviors_df.empty:                                         # Check whether the behaviors dataset was downloaded successfully and is not empty.
    behaviors_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "behaviors.parquet"), index=False)  # Save the local combined behaviors file for the next training step.

if not vectors_df.empty:                                           # Check whether the user vectors dataset was downloaded successfully and is not empty.
    vectors_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "user_vectors.parquet"), index=False)  # Save the local combined user vectors file for the next training step.

if not logs_df.empty:                                              # Check whether the processed user logs dataset was downloaded successfully and is not empty.
    logs_df.to_parquet(os.path.join(LOCAL_OUTPUT_DIR, "user_logs.parquet"), index=False)  # Save the local combined user logs file for the next training step.

print("\nSaved local files:")                                      # Print a heading before listing the files created in the local training_data folder.
for filename in os.listdir(LOCAL_OUTPUT_DIR):                      # Loop through every file saved in the local training_data folder.
    print(f"- {filename}")                                         # Print the file name to confirm the expected outputs were created.

print("\nPhase 5 Step 1 complete.")                                # Print the final success message for this phase step.
