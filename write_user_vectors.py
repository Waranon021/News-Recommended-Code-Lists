# write_user_vectors.py

import os                                                                 # Import os so the script can build temporary local file paths safely.
import json                                                               # Import json so the script can read raw simulated user log files from S3.
import tempfile                                                           # Import tempfile so parquet files can be downloaded into a temporary folder on Windows.
from decimal import Decimal                                               # Import Decimal because DynamoDB stores numeric values more safely in Decimal format.

import boto3                                                              # Import boto3 so the script can read data from S3 and write user vectors into DynamoDB.
import pandas as pd                                                       # Import pandas so the script can load parquet files, merge data, and clean list columns.


PROC_BUCKET = "news-recommending-processed-st125934"                      # Name of the processed S3 bucket that stores users/vectors parquet files.
RAW_BUCKET = "news-recommending-raw-st125934"                             # Name of the raw S3 bucket that stores simulated user log JSON files.
DYNAMO_TABLE = "user-vectors"                                             # Name of the DynamoDB table where final user vectors will be written.
AWS_REGION = "ap-southeast-7"                                             # AWS region for your current account, which is Thailand right now.
CATEGORY_COLS = ["business", "entertainment", "technology", "health", "science"]  # Fixed category columns expected in the user vector table.


s3 = boto3.client("s3", region_name=AWS_REGION)                           # Create one S3 client that all helper functions will use to read parquet and JSON files.
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)             # Create one DynamoDB resource that will be used to write items into the target table.
table = dynamodb.Table(DYNAMO_TABLE)                                      # Get the specific DynamoDB table object for user vectors.


def to_decimal_int(value):                                                # Define a helper function to convert numeric values into DynamoDB-safe Decimal integers.
    if value is None:                                                     # Check whether the input value is completely missing.
        return Decimal("0")                                               # Return Decimal zero if the value is missing.

    try:                                                                  # Start a try block so pd.isna can be tested safely.
        if pd.isna(value):                                                # Check whether the value is NaN or missing according to pandas.
            return Decimal("0")                                           # Return Decimal zero if the value is NaN.
    except Exception:                                                     # Catch any error from pd.isna on unsupported types.
        pass                                                              # Ignore the error and continue to the integer conversion step.

    return Decimal(str(int(value)))                                       # Convert the value to int, then string, then Decimal for safe DynamoDB storage.


def normalize_click_list(value):                                          # Define a helper function that converts many possible click-list formats into a clean Python list.
    if value is None:                                                     # Check whether the value is completely missing.
        return []                                                         # Return an empty list if there is no value.

    if isinstance(value, list):                                           # Check whether the value is already a Python list.
        return [str(x) for x in value if x is not None]                   # Keep the list but convert every element to string and remove None values.

    if isinstance(value, (tuple, set)):                                   # Check whether the value is a tuple or set instead of a list.
        return [str(x) for x in list(value) if x is not None]             # Convert it into a normal Python list of strings.

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):  # Check whether the value is a numpy/pyarrow/pandas array-like object.
        try:                                                              # Start a try block so unusual array formats do not crash the script.
            converted = value.tolist()                                    # Convert the array-like object into a normal Python object first.

            if converted is None:                                         # Check whether the converted value is still empty.
                return []                                                 # Return an empty list if the converted result is None.

            if isinstance(converted, list):                               # Check whether the converted value is already a list.
                return [str(x) for x in converted if x is not None]       # Return the cleaned list of strings.

            if isinstance(converted, (tuple, set)):                       # Check whether the converted value is a tuple or set.
                return [str(x) for x in list(converted) if x is not None] # Convert the tuple/set into a cleaned list of strings.

            return [str(converted)]                                       # If the converted value is a single scalar, wrap it into a one-item list.
        except Exception:                                                 # Catch any conversion error for strange array-like objects.
            pass                                                          # Ignore the error and continue to the later fallback logic.

    if isinstance(value, str):                                            # Check whether the value is a normal string.
        cleaned = value.strip()                                           # Strip whitespace from the string.
        if cleaned == "" or cleaned.lower() == "nan":                     # Check whether the cleaned string is empty or just text NaN.
            return []                                                     # Return an empty list for empty or NaN-like strings.
        return [cleaned]                                                  # Otherwise wrap the cleaned string into a one-item list.

    try:                                                                  # Start a try block for scalar NaN checking.
        if pd.isna(value):                                                # Check whether the scalar value is NaN.
            return []                                                     # Return an empty list if the scalar value is NaN.
    except Exception:                                                     # Catch any error from pd.isna on unsupported types.
        pass                                                              # Ignore the error and continue to the final fallback.

    return [str(value)]                                                   # Use a final fallback that converts the value into a one-item list of string.


def list_parquet_files(bucket, prefix):                                   # Define a helper function that lists every parquet file under a given S3 folder prefix.
    paginator = s3.get_paginator("list_objects_v2")                       # Create a paginator because one S3 folder may contain many parquet files.
    parquet_keys = []                                                     # Create an empty list to store the matching parquet file keys.

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):         # Loop through every result page returned by S3 for the selected bucket and prefix.
        for obj in page.get("Contents", []):                              # Loop through each file object inside the current result page.
            if obj["Key"].endswith(".parquet"):                           # Keep only files that end with .parquet because those are the processed datasets needed.
                parquet_keys.append(obj["Key"])                           # Add the parquet file path into the list.

    return parquet_keys                                                   # Return the full list of parquet file paths found in that S3 folder.


def load_vectors_from_s3(bucket, prefix):                                 # Define a helper function that downloads all users/vectors parquet files from S3 and combines them.
    parquet_files = list_parquet_files(bucket, prefix)                    # Call the previous helper to get every parquet file under users/vectors/.

    if not parquet_files:                                                 # Check whether no parquet files were found at all.
        raise ValueError(f"No parquet files found in s3://{bucket}/{prefix}")  # Stop with a clear error if the folder is empty.

    print(f"Found {len(parquet_files)} parquet files in s3://{bucket}/{prefix}")  # Print how many parquet parts were found in S3.

    tmp_dir = tempfile.gettempdir()                                       # Get the system temporary folder path so the files can be downloaded locally first.
    dfs = []                                                              # Create an empty list to hold each parquet part after loading it into pandas.

    for key in parquet_files:                                             # Loop through each parquet file path found in S3.
        local_path = os.path.join(tmp_dir, f"vectors_{os.path.basename(key)}")  # Build a temporary local path using the parquet file name.
        s3.download_file(bucket, key, local_path)                         # Download the parquet file from S3 to the temporary local path.
        df = pd.read_parquet(local_path)                                  # Read the downloaded parquet file into a pandas DataFrame.
        dfs.append(df)                                                    # Add that dataframe into the list of parquet parts.

    vectors_df = pd.concat(dfs, ignore_index=True)                        # Combine all parquet parts into one full user vector dataframe and reset row numbering.

    print(f"User vectors loaded: {len(vectors_df)} users")                # Print the total number of user vector rows loaded from parquet.
    print(f"Columns: {list(vectors_df.columns)}")                         # Print the dataframe column names to verify the structure.

    return vectors_df                                                     # Return the combined user vector dataframe.


def load_raw_logs_from_s3(bucket, prefix):                                # Define a helper function that reads all raw simulated log JSON files from S3.
    paginator = s3.get_paginator("list_objects_v2")                       # Create a paginator because one S3 folder may contain many JSON files.
    logs = []                                                             # Create an empty list to store the decoded log records.

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):         # Loop through every result page returned by S3 for the selected raw log prefix.
        for obj in page.get("Contents", []):                              # Loop through each file object inside the current result page.
            if obj["Key"].endswith(".json"):                              # Keep only files that end with .json because those are the raw simulated logs.
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])       # Download the JSON file content directly from S3.
                payload = json.loads(resp["Body"].read())                 # Read the S3 object body and decode it from JSON text into Python objects.

                if isinstance(payload, list):                             # Check whether the payload is a list of interaction records.
                    logs.extend(payload)                                  # Add all interaction records into the combined logs list.

    logs_df = pd.DataFrame(logs)                                          # Convert the final combined log list into a pandas DataFrame.
    print(f"Logs loaded: {len(logs_df)} interactions")                    # Print how many raw simulated interactions were loaded.
    return logs_df                                                        # Return the logs dataframe.


def build_recent_clicks(logs_df):                                         # Define a helper function that keeps the latest 10 clicked article IDs per user from raw logs.
    if logs_df.empty:                                                     # Check whether the logs dataframe is completely empty.
        return pd.DataFrame(columns=["user_id", "recent_clicks"])         # Return an empty dataframe with the expected columns.

    required_cols = {"user_id", "article_id", "action", "timestamp"}      # Define the minimum columns needed for building recent click history.
    if not required_cols.issubset(set(logs_df.columns)):                  # Check whether any required column is missing from the logs dataframe.
        return pd.DataFrame(columns=["user_id", "recent_clicks"])         # Return an empty dataframe if the structure is incomplete.

    clicks_df = logs_df[logs_df["action"] == "click"].copy()              # Keep only log rows where the user actually clicked an article.

    if clicks_df.empty:                                                   # Check whether there are no click rows at all.
        return pd.DataFrame(columns=["user_id", "recent_clicks"])         # Return an empty dataframe if no click interactions exist.

    clicks_df = clicks_df.sort_values("timestamp", ascending=False)       # Sort clicks from newest to oldest so head(10) keeps the most recent ones.

    recent_clicks = (                                                     # Start a grouped transformation pipeline for recent clicks.
        clicks_df.groupby("user_id")["article_id"]                        # Group clicks by user and focus only on the article_id column.
        .apply(lambda x: [str(v) for v in list(x.head(10))])              # Keep the top 10 newest article IDs for each user and convert them to strings.
        .reset_index()                                                    # Turn the grouped result back into a normal dataframe.
        .rename(columns={"article_id": "recent_clicks"})                  # Rename the output column from article_id to recent_clicks.
    )

    return recent_clicks                                                  # Return the final dataframe containing user_id and recent_clicks.


def compute_top_category(category_prefs):                                 # Define a helper function that finds the strongest category for one user.
    int_prefs = {k: int(v) for k, v in category_prefs.items()}            # Convert the Decimal values in category_preferences into normal integers first.

    if sum(int_prefs.values()) == 0:                                      # Check whether all category counts are zero.
        return "unknown"                                                  # Return unknown if the user has no clicks at all.

    return max(int_prefs, key=int_prefs.get)                              # Otherwise return the category name with the largest count.


print("Reading user vectors from S3...")                                  # Print a start message before loading processed user vectors.
vectors_df = load_vectors_from_s3(PROC_BUCKET, "users/vectors/")          # Load all processed user vector parquet files from S3.

print("Reading simulated logs...")                                        # Print a message before loading raw simulated user logs.
logs_df = load_raw_logs_from_s3(RAW_BUCKET, "user-logs/")                 # Load all raw simulated user log JSON files from S3.

recent_clicks_df = build_recent_clicks(logs_df)                           # Build the most recent 10 clicked article IDs per user from the raw logs.

merged = vectors_df.merge(                                                # Merge the ETL-generated user vectors with the raw-log recent click history.
    recent_clicks_df,                                                     # Use the recent_clicks dataframe as the right-side merge input.
    on="user_id",                                                         # Merge both dataframes by user_id.
    how="left",                                                           # Keep all ETL user vectors even if some users have no raw recent click list.
    suffixes=("_etl", "_raw")                                             # Add suffixes so both recent_clicks columns can exist safely if needed.
)

if "recent_clicks_etl" in merged.columns and "recent_clicks_raw" in merged.columns:  # Check whether both ETL and raw recent_clicks columns exist after merge.
    merged["recent_clicks"] = merged["recent_clicks_raw"].where(          # Prefer the raw recent_clicks column when it has a value.
        merged["recent_clicks_raw"].notna(),                              # Keep raw recent clicks when they are not null.
        merged["recent_clicks_etl"]                                       # Otherwise fall back to the ETL-generated recent click list.
    )
elif "recent_clicks_raw" in merged.columns:                               # Check whether only the raw recent_clicks column exists.
    merged["recent_clicks"] = merged["recent_clicks_raw"]                 # Use the raw recent_clicks column directly.
elif "recent_clicks_etl" in merged.columns:                               # Check whether only the ETL recent_clicks column exists.
    merged["recent_clicks"] = merged["recent_clicks_etl"]                 # Use the ETL recent_clicks column directly.
else:                                                                     # Handle the case where neither recent_clicks column exists.
    merged["recent_clicks"] = [[] for _ in range(len(merged))]            # Create an empty list for every row so the downstream code still works.

merged["recent_clicks"] = merged["recent_clicks"].apply(normalize_click_list)  # Normalize recent_clicks into clean Python lists of strings.

for col in CATEGORY_COLS:                                                 # Loop through every expected category column.
    if col not in merged.columns:                                         # Check whether the category column is missing in the merged dataframe.
        merged[col] = 0                                                   # Create the missing category column and fill it with 0.

print(f"Merged rows: {len(merged)}")                                      # Print the total number of merged user rows.
print("Writing user vectors to DynamoDB...")                              # Print a message to know the script is starting the DynamoDB write step.

success_count = 0                                                         # Create a counter to track how many user vector items were written successfully.
error_count = 0                                                           # Create a counter to track how many rows failed during writing.

with table.batch_writer(overwrite_by_pkeys=["user_id"]) as batch:         # Open a DynamoDB batch writer so items can be written more efficiently in groups.
    for _, row in merged.iterrows():                                      # Loop through each merged user row one by one.
        try:                                                              # Start a try block so one bad row does not crash the entire script.
            category_prefs = {}                                           # Create an empty dictionary that will store category counts for this user.
            for col in CATEGORY_COLS:                                     # Loop through every fixed category.
                category_prefs[col] = to_decimal_int(row[col])            # Convert the category count into Decimal and save it into the dictionary.

            row_top_category = row.get("top_category", None)              # Try to read the top_category value that already exists in the merged row.
            if pd.notna(row_top_category) and str(row_top_category).strip() != "":  # Check whether top_category already has a valid non-empty value.
                top_category = str(row_top_category)                      # Use the existing top_category value from the row.
            else:                                                         # Handle the case where top_category is missing or empty.
                top_category = compute_top_category(category_prefs)       # Recompute top_category from the category_preferences dictionary.

            row_total_clicks = row.get("total_clicks", None)              # Try to read the total_clicks value that already exists in the merged row.
            if row_total_clicks is not None:                              # Check whether total_clicks exists at all.
                try:                                                      # Start a try block so strange total_clicks values do not crash the script.
                    if pd.notna(row_total_clicks):                        # Check whether total_clicks is a valid non-null value.
                        total_clicks = to_decimal_int(row_total_clicks)   # Convert the existing total_clicks value into Decimal.
                    else:                                                 # Handle the case where total_clicks exists but is null.
                        total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))  # Rebuild total_clicks by summing category counts.
                except Exception:                                         # Catch conversion errors for unusual total_clicks values.
                    total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))  # Rebuild total_clicks from the category counts if conversion fails.
            else:                                                         # Handle the case where total_clicks does not exist at all.
                total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))  # Build total_clicks by summing all category counts.

            item = {                                                      # Build one final DynamoDB item for this user.
                "user_id": str(row["user_id"]),                           # Save the user_id as the DynamoDB partition key.
                "category_preferences": category_prefs,                   # Save the category count dictionary.
                "top_category": top_category,                             # Save the user's strongest category.
                "recent_clicks": [str(x) for x in row["recent_clicks"]],  # Save the cleaned list of recent clicked article IDs.
                "total_clicks": total_clicks,                             # Save the total click count for this user.
            }

            batch.put_item(Item=item)                                     # Write the item into DynamoDB using the batch writer.
            success_count += 1                                            # Increase the success counter after a successful write.

        except Exception as e:                                            # Catch any row-level error during DynamoDB writing.
            error_count += 1                                              # Increase the error counter when a row fails.
            print(f"Error on row: {e}")                                   # Print the error but continue processing the rest of the rows.

print(f"Written {success_count} user vectors to DynamoDB.")               # Print the total number of successfully written DynamoDB items.
print(f"Errors: {error_count}")                                           # Print the total number of row-level errors.
print("Phase 4 user vectors complete.")                                   # Print the final success message for this phase step.
