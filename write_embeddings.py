# write_embeddings.py

# import os for file path handling
import os

# import tempfile so the script works on Windows without hardcoding /tmp
import tempfile

# import hashlib to create a stable article_id from each article URL
import hashlib

# import boto3 to read from S3 and write to DynamoDB
import boto3

# import pandas to read and combine parquet files
import pandas as pd

# import Decimal because DynamoDB does not like normal Python floats
from decimal import Decimal

# import the sentence-transformer model used in your Phase 4 plan
from sentence_transformers import SentenceTransformer


# ------------------------------
# Config
# ------------------------------

# processed S3 bucket that contains your parquet files
PROC_BUCKET = "news-recommender-processed-st125934"

# folder inside the processed bucket where article parquet files are stored
ARTICLES_PREFIX = "articles/"

# DynamoDB table name for article embeddings
DYNAMO_TABLE = "article-embeddings"

# AWS region used throughout your project
AWS_REGION = "us-east-1"

# cap the number of unique articles to keep runtime manageable
MAX_ARTICLES = 5000

# embedding model from your friend's Phase 4 instructions
MODEL_NAME = "all-MiniLM-L6-v2"


# ------------------------------
# Helper functions
# ------------------------------

# create one S3 client for reading parquet files
s3 = boto3.client("s3", region_name=AWS_REGION)

# create one DynamoDB resource for writing items
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

# get the DynamoDB table object
table = dynamodb.Table(DYNAMO_TABLE)


# convert numeric values to Decimal for DynamoDB
def to_decimal(value):
    # round floats slightly so the stored vectors are smaller and cleaner
    return Decimal(str(round(float(value), 6)))


# create a stable article_id from the article URL if possible
def make_article_id(row):
    # try to use URL first because it is the best identifier in your current dataset
    url = str(row.get("url", "")).strip()

    # if URL exists, hash it to create a fixed key
    if url:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    # fallback if URL is missing
    fallback_text = "||".join(
        [
            str(row.get("title", "")).strip(),
            str(row.get("source_name", row.get("source", ""))).strip(),
            str(row.get("publishedAt", row.get("pub_date", ""))).strip(),
        ]
    )

    # hash the fallback text too
    return hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()


# list all parquet files under the articles prefix
def list_parquet_files(bucket, prefix):
    # use paginator in case there are many files
    paginator = s3.get_paginator("list_objects_v2")

    # keep all parquet keys here
    parquet_keys = []

    # loop through all pages
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        # get all objects from the current page
        for obj in page.get("Contents", []):
            # keep only parquet files
            if obj["Key"].endswith(".parquet"):
                parquet_keys.append(obj["Key"])

    # return the full list of parquet keys
    return parquet_keys


# download all parquet files locally and combine into one DataFrame
def load_articles_from_s3(bucket, prefix):
    # find all parquet files first
    parquet_files = list_parquet_files(bucket, prefix)

    # fail early if none were found
    if not parquet_files:
        raise ValueError(f"No parquet files found in s3://{bucket}/{prefix}")

    # print how many parquet files were found
    print(f"Found {len(parquet_files)} parquet files in s3://{bucket}/{prefix}")

    # get a temp directory that works on Windows and Linux
    tmp_dir = tempfile.gettempdir()

    # store each parquet dataframe here
    dfs = []

    # loop through every parquet key
    for key in parquet_files:
        # build a local temp file path
        local_path = os.path.join(tmp_dir, os.path.basename(key))

        # download the parquet file from S3
        s3.download_file(bucket, key, local_path)

        # read the parquet file into pandas
        df = pd.read_parquet(local_path)

        # append to the list
        dfs.append(df)

    # combine all parquet parts into one dataframe
    articles_df = pd.concat(dfs, ignore_index=True)

    # print total rows before cleaning
    print(f"Loaded {len(articles_df)} raw article rows from parquet")

    # return the combined dataframe
    return articles_df


# build the text used for embedding from title + abstract/content
def build_text(row):
    # get title safely
    title = str(row.get("title", "")).strip()

    # your dataframe may have abstract, description, or content depending on ETL shape
    abstract = str(row.get("abstract", "")).strip()
    description = str(row.get("description", "")).strip()
    content = str(row.get("content", "")).strip()

    # choose the richest available text after title
    body = abstract or description or content

    # combine title and body
    return f"{title} {body}".strip()


# ------------------------------
# Main workflow
# ------------------------------

# load the sentence-transformers model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully")

# load article parquet files from S3
articles_df = load_articles_from_s3(PROC_BUCKET, ARTICLES_PREFIX)

# keep only rows with a non-empty title
articles_df = articles_df[
    articles_df["title"].notna() & (articles_df["title"].astype(str).str.strip() != "")
].copy()

# print row count after dropping empty titles
print(f"Rows after dropping empty titles: {len(articles_df)}")

# deduplicate by URL first if URL column exists
if "url" in articles_df.columns:
    # drop missing/blank URLs before deduplication
    articles_df["url"] = articles_df["url"].fillna("").astype(str).str.strip()

    # split rows with URL and without URL
    with_url = articles_df[articles_df["url"] != ""].copy()
    without_url = articles_df[articles_df["url"] == ""].copy()

    # deduplicate rows that have URLs
    before_url_dedup = len(with_url)
    with_url = with_url.drop_duplicates(subset=["url"]).copy()
    after_url_dedup = len(with_url)

    # combine back with rows that had no URL
    articles_df = pd.concat([with_url, without_url], ignore_index=True)

    # print dedup stats
    print(f"Rows with URL before dedup: {before_url_dedup}")
    print(f"Rows with URL after dedup : {after_url_dedup}")
    print(f"Total rows after URL dedup: {len(articles_df)}")

# if title still exists, do a second light dedup on title for rows without URL
articles_df["title"] = articles_df["title"].astype(str).str.strip()
articles_df = articles_df.drop_duplicates(subset=["title", "url"]).copy()

# print row count after final dedup
print(f"Rows after final dedup: {len(articles_df)}")

# limit to MAX_ARTICLES for runtime control
if len(articles_df) > MAX_ARTICLES:
    articles_df = articles_df.head(MAX_ARTICLES).copy()
    print(f"Trimmed to {MAX_ARTICLES} articles for embedding generation")

# create stable article IDs
articles_df["article_id"] = articles_df.apply(make_article_id, axis=1)

# build embedding text
articles_df["embedding_text"] = articles_df.apply(build_text, axis=1)

# keep only rows with non-empty embedding text
articles_df = articles_df[
    articles_df["embedding_text"].notna()
    & (articles_df["embedding_text"].astype(str).str.strip() != "")
].copy()

# print final row count to embed
print(f"Final articles to embed: {len(articles_df)}")

# convert embedding text to a Python list
texts = articles_df["embedding_text"].tolist()

# generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Embeddings generated with shape: {embeddings.shape}")

# write each article + vector into DynamoDB
print(f"Writing items to DynamoDB table: {DYNAMO_TABLE}")

# counters for progress reporting
success_count = 0
error_count = 0

# batch_writer automatically handles DynamoDB batching
with table.batch_writer(overwrite_by_pkeys=["article_id"]) as batch:
    # iterate row-by-row together with each embedding vector
    for (_, row), vector in zip(articles_df.iterrows(), embeddings):
        try:
            # convert vector to DynamoDB-safe Decimal list
            embedding_list = [to_decimal(v) for v in vector.tolist()]

            # build the DynamoDB item
            item = {
                "article_id": row["article_id"],
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "category": str(row.get("category", "")),
                "source": str(row.get("source_name", row.get("source", ""))),
                "abstract": str(row.get("abstract", row.get("description", ""))),
                "pub_date": str(row.get("publishedAt", row.get("pub_date", ""))),
                "fetched_at": str(row.get("fetched_at", "")),
                "data_source": str(row.get("data_source", "newsapi")),
                "embedding": embedding_list,
                "embedding_dim": int(len(embedding_list)),
            }

            # write the item
            batch.put_item(Item=item)

            # increment success counter
            success_count += 1

        except Exception as e:
            # increment error counter
            error_count += 1

            # print the error but continue
            print(f"Error writing row to DynamoDB: {e}")

# final summary
print("Done.")
print(f"Successfully wrote: {success_count}")
print(f"Errors: {error_count}")