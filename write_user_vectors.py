# write_user_vectors.py

import os
import json
import tempfile
from decimal import Decimal

import boto3
import pandas as pd


# ------------------------------
# Config
# ------------------------------
PROC_BUCKET = "news-recommender-processed-st125934"
RAW_BUCKET = "news-recommender-raw-st125934"
DYNAMO_TABLE = "user-vectors"
AWS_REGION = "us-east-1"
CATEGORY_COLS = ["business", "entertainment", "technology", "health", "science"]


# ------------------------------
# AWS clients
# ------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMO_TABLE)


# ------------------------------
# Helper functions
# ------------------------------
def to_decimal_int(value):
    """Convert numeric values into DynamoDB-safe Decimal integers."""
    if value is None:
        return Decimal("0")

    try:
        if pd.isna(value):
            return Decimal("0")
    except Exception:
        pass

    return Decimal(str(int(value)))


def normalize_click_list(value):
    """Convert different array-like / scalar forms into a clean Python list of strings."""
    # None -> []
    if value is None:
        return []

    # already a list
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]

    # tuple/set -> list
    if isinstance(value, (tuple, set)):
        return [str(x) for x in list(value) if x is not None]

    # numpy / pyarrow / pandas array-like objects
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()

            if converted is None:
                return []

            if isinstance(converted, list):
                return [str(x) for x in converted if x is not None]

            if isinstance(converted, (tuple, set)):
                return [str(x) for x in list(converted) if x is not None]

            return [str(converted)]
        except Exception:
            pass

    # strings
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "" or cleaned.lower() == "nan":
            return []
        return [cleaned]

    # scalar NaN safely
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    # final fallback
    return [str(value)]


def list_parquet_files(bucket, prefix):
    """List all parquet files under the given S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    parquet_keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_keys.append(obj["Key"])

    return parquet_keys


def load_vectors_from_s3(bucket, prefix):
    """Load all parquet files under users/vectors into one DataFrame."""
    parquet_files = list_parquet_files(bucket, prefix)

    if not parquet_files:
        raise ValueError(f"No parquet files found in s3://{bucket}/{prefix}")

    print(f"Found {len(parquet_files)} parquet files in s3://{bucket}/{prefix}")

    tmp_dir = tempfile.gettempdir()
    dfs = []

    for key in parquet_files:
        local_path = os.path.join(tmp_dir, f"vectors_{os.path.basename(key)}")
        s3.download_file(bucket, key, local_path)
        df = pd.read_parquet(local_path)
        dfs.append(df)

    vectors_df = pd.concat(dfs, ignore_index=True)

    print(f"User vectors loaded: {len(vectors_df)} users")
    print(f"Columns: {list(vectors_df.columns)}")

    return vectors_df


def load_raw_logs_from_s3(bucket, prefix):
    """Load all raw JSON simulated user logs from S3 into one DataFrame."""
    paginator = s3.get_paginator("list_objects_v2")
    logs = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                resp = s3.get_object(Bucket=bucket, Key=obj["Key"])
                payload = json.loads(resp["Body"].read())

                if isinstance(payload, list):
                    logs.extend(payload)

    logs_df = pd.DataFrame(logs)
    print(f"Logs loaded: {len(logs_df)} interactions")
    return logs_df


def build_recent_clicks(logs_df):
    """Build last 10 clicked article_ids per user from raw logs."""
    if logs_df.empty:
        return pd.DataFrame(columns=["user_id", "recent_clicks"])

    required_cols = {"user_id", "article_id", "action", "timestamp"}
    if not required_cols.issubset(set(logs_df.columns)):
        return pd.DataFrame(columns=["user_id", "recent_clicks"])

    clicks_df = logs_df[logs_df["action"] == "click"].copy()

    if clicks_df.empty:
        return pd.DataFrame(columns=["user_id", "recent_clicks"])

    clicks_df = clicks_df.sort_values("timestamp", ascending=False)

    recent_clicks = (
        clicks_df.groupby("user_id")["article_id"]
        .apply(lambda x: [str(v) for v in list(x.head(10))])
        .reset_index()
        .rename(columns={"article_id": "recent_clicks"})
    )

    return recent_clicks


def compute_top_category(category_prefs):
    """Find top category from category preference dictionary."""
    int_prefs = {k: int(v) for k, v in category_prefs.items()}

    if sum(int_prefs.values()) == 0:
        return "unknown"

    return max(int_prefs, key=int_prefs.get)


# ------------------------------
# Main workflow
# ------------------------------
print("Reading user vectors from S3...")
vectors_df = load_vectors_from_s3(PROC_BUCKET, "users/vectors/")

print("Reading simulated logs...")
logs_df = load_raw_logs_from_s3(RAW_BUCKET, "user-logs/")

recent_clicks_df = build_recent_clicks(logs_df)

# merge ETL vectors with raw-log recent clicks
merged = vectors_df.merge(
    recent_clicks_df,
    on="user_id",
    how="left",
    suffixes=("_etl", "_raw")
)

# choose which recent_clicks to keep
if "recent_clicks_etl" in merged.columns and "recent_clicks_raw" in merged.columns:
    merged["recent_clicks"] = merged["recent_clicks_raw"].where(
        merged["recent_clicks_raw"].notna(),
        merged["recent_clicks_etl"]
    )
elif "recent_clicks_raw" in merged.columns:
    merged["recent_clicks"] = merged["recent_clicks_raw"]
elif "recent_clicks_etl" in merged.columns:
    merged["recent_clicks"] = merged["recent_clicks_etl"]
else:
    merged["recent_clicks"] = [[] for _ in range(len(merged))]

# normalize recent_clicks to plain Python lists
merged["recent_clicks"] = merged["recent_clicks"].apply(normalize_click_list)

# ensure category columns exist
for col in CATEGORY_COLS:
    if col not in merged.columns:
        merged[col] = 0

print(f"Merged rows: {len(merged)}")
print("Writing user vectors to DynamoDB...")

success_count = 0
error_count = 0

with table.batch_writer(overwrite_by_pkeys=["user_id"]) as batch:
    for _, row in merged.iterrows():
        try:
            category_prefs = {}
            for col in CATEGORY_COLS:
                category_prefs[col] = to_decimal_int(row[col])

            row_top_category = row.get("top_category", None)
            if pd.notna(row_top_category) and str(row_top_category).strip() != "":
                top_category = str(row_top_category)
            else:
                top_category = compute_top_category(category_prefs)

            row_total_clicks = row.get("total_clicks", None)
            if row_total_clicks is not None:
                try:
                    if pd.notna(row_total_clicks):
                        total_clicks = to_decimal_int(row_total_clicks)
                    else:
                        total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))
                except Exception:
                    total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))
            else:
                total_clicks = Decimal(str(sum(int(v) for v in category_prefs.values())))

            item = {
                "user_id": str(row["user_id"]),
                "category_preferences": category_prefs,
                "top_category": top_category,
                "recent_clicks": [str(x) for x in row["recent_clicks"]],
                "total_clicks": total_clicks,
            }

            batch.put_item(Item=item)
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error on row: {e}")

print(f"Written {success_count} user vectors to DynamoDB.")
print(f"Errors: {error_count}")
print("Phase 4 user vectors complete.")