import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# get the Glue job name from runtime arguments
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

# start Spark / Glue contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# raw and processed bucket paths
RAW_BUCKET = "s3://news-recommender-raw-st125934/"
PROCESSED_BUCKET = "s3://news-recommender-processed-st125934/"

# fixed project categories used in your simulated logs and later user vectors
FIXED_CATEGORIES = ["business", "entertainment", "technology", "health", "science"]


# -----------------------------
# 1) LOAD AND CLEAN NEWSAPI
# -----------------------------

print("Loading NewsAPI JSON...")

# read every JSON file under raw/newsapi/
newsapi_df = spark.read.option("multiline", "true").json(RAW_BUCKET + "newsapi/")

# flatten nested source object if present
if "source" in newsapi_df.columns:
    newsapi_df = newsapi_df.withColumn("source_name", F.col("source.name")).drop("source")
else:
    newsapi_df = newsapi_df.withColumn("source_name", F.lit("unknown"))

# build a clean standard article schema for NewsAPI
newsapi_std = newsapi_df.select(
    F.sha2(
        F.lower(
            F.coalesce(
                F.trim(F.col("url")),
                F.concat_ws(
                    "||",
                    F.coalesce(F.col("title"), F.lit("")),
                    F.coalesce(F.col("publishedAt"), F.lit("")),
                    F.coalesce(F.col("source_name"), F.lit(""))
                )
            )
        ),
        256
    ).alias("article_id"),
    F.coalesce(F.trim(F.col("url")), F.lit("")).alias("url"),
    F.coalesce(F.col("title"), F.lit("")).alias("title"),
    F.coalesce(F.col("description"), F.lit("")).alias("abstract"),
    F.coalesce(F.col("content"), F.lit("")).alias("content"),
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),
    F.coalesce(F.col("source_name"), F.lit("unknown")).alias("source"),
    F.coalesce(F.col("publishedAt"), F.lit("")).alias("pub_date"),
    F.coalesce(F.col("fetched_at"), F.lit("")).alias("fetched_at"),
    F.lit("newsapi").alias("data_source")
)

# use URL when available as the dedup key; otherwise fall back to article_id
newsapi_std = newsapi_std.withColumn(
    "article_key",
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))
)

# keep the latest version of each NewsAPI article
newsapi_window = Window.partitionBy("article_key").orderBy(
    F.col("fetched_at").desc_nulls_last(),
    F.col("pub_date").desc_nulls_last()
)

newsapi_dedup = (
    newsapi_std
    .withColumn("rn", F.row_number().over(newsapi_window))
    .filter(F.col("rn") == 1)
    .drop("rn", "article_key")
)

print("NewsAPI rows after dedup:")
print(newsapi_dedup.count())


# -----------------------------
# 2) LOAD AND CLEAN MIND NEWS
# -----------------------------

print("Loading MIND news TSV...")

# Spark reads all files inside mind/news/ including train_news.tsv and val_news.tsv
mind_news_raw = (
    spark.read
    .option("sep", "\t")
    .option("header", "false")
    .csv(RAW_BUCKET + "mind/news/")
)

# assign standard MIND news columns
mind_news_df = mind_news_raw.toDF(
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities"
)

# build the same article schema for MIND
mind_news_std = mind_news_df.select(
    F.sha2(
        F.lower(
            F.coalesce(
                F.trim(F.col("url")),
                F.trim(F.col("news_id"))
            )
        ),
        256
    ).alias("article_id"),
    F.coalesce(
        F.trim(F.col("url")),
        F.concat(F.lit("mind://"), F.col("news_id"))
    ).alias("url"),
    F.coalesce(F.col("title"), F.lit("")).alias("title"),
    F.coalesce(F.col("abstract"), F.lit("")).alias("abstract"),
    F.lit("").alias("content"),
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),
    F.lit("microsoft_mind").alias("source"),
    F.lit("").alias("pub_date"),
    F.lit("").alias("fetched_at"),
    F.lit("mind").alias("data_source")
)

# use URL or article_id as final dedup key
mind_news_std = mind_news_std.withColumn(
    "article_key",
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))
)

# deduplicate MIND itself
mind_window = Window.partitionBy("article_key").orderBy(F.col("article_id"))
mind_news_dedup = (
    mind_news_std
    .withColumn("rn", F.row_number().over(mind_window))
    .filter(F.col("rn") == 1)
    .drop("rn", "article_key")
)

print("MIND news rows after dedup:")
print(mind_news_dedup.count())


# -----------------------------
# 3) COMBINE ALL ARTICLES
# -----------------------------

print("Combining NewsAPI + MIND articles...")

common_article_cols = [
    "article_id",
    "url",
    "title",
    "abstract",
    "content",
    "category",
    "source",
    "pub_date",
    "fetched_at",
    "data_source"
]

articles_df = newsapi_dedup.select(common_article_cols).unionByName(
    mind_news_dedup.select(common_article_cols)
)

# deduplicate again across both sources
articles_df = articles_df.withColumn(
    "article_key",
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))
)

combined_window = Window.partitionBy("article_key").orderBy(
    F.col("fetched_at").desc_nulls_last(),
    F.col("pub_date").desc_nulls_last(),
    F.col("data_source").desc()
)

articles_df = (
    articles_df
    .withColumn("rn", F.row_number().over(combined_window))
    .filter(F.col("rn") == 1)
    .drop("rn", "article_key")
)

print("Final combined article rows:")
print(articles_df.count())

# write final articles parquet
articles_df.write.mode("overwrite").parquet(PROCESSED_BUCKET + "articles/")


# -----------------------------
# 4) LOAD AND CLEAN SIMULATED USER LOGS
# -----------------------------

print("Loading simulated user logs JSON...")

logs_df = spark.read.option("multiline", "true").json(RAW_BUCKET + "user-logs/")

logs_clean = logs_df.select(
    F.coalesce(F.col("user_id"), F.lit("")).alias("user_id"),
    F.coalesce(F.col("article_id"), F.lit("")).alias("article_id"),
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),
    F.lower(F.coalesce(F.col("action"), F.lit("view"))).alias("action"),
    F.coalesce(F.col("session_id"), F.lit("")).alias("session_id"),
    F.coalesce(F.col("timestamp"), F.lit("")).alias("timestamp")
)

# write cleaned user logs parquet
logs_clean.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/logs/")


# -----------------------------
# 5) LOAD AND CLEAN MIND BEHAVIORS
# -----------------------------

print("Loading MIND behaviors TSV...")

# Spark reads all files inside mind/behaviors/ including train_behaviors.tsv and val_behaviors.tsv
mind_behaviors_raw = (
    spark.read
    .option("sep", "\t")
    .option("header", "false")
    .csv(RAW_BUCKET + "mind/behaviors/")
)

# assign MIND behavior columns
mind_behaviors_df = mind_behaviors_raw.toDF(
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions"
)

# explode each impression entry like "N12345-1"
behaviors_exploded = (
    mind_behaviors_df
    .withColumn("impression_item", F.explode(F.split(F.coalesce(F.col("impressions"), F.lit("")), " ")))
    .filter(F.trim(F.col("impression_item")) != "")
)

# parse article reference and click label from each impression item
behaviors_parsed = (
    behaviors_exploded
    .withColumn("article_ref", F.regexp_extract(F.col("impression_item"), r"^(.*)-[01]$", 1))
    .withColumn("clicked", F.regexp_extract(F.col("impression_item"), r".*-(\d)$", 1).cast("int"))
)

# join to MIND news to recover category
mind_news_lookup = mind_news_df.select(
    F.col("news_id").alias("article_ref"),
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category")
).dropDuplicates(["article_ref"])

behaviors_clean = (
    behaviors_parsed
    .join(mind_news_lookup, on="article_ref", how="left")
    .select(
        F.coalesce(F.col("user_id"), F.lit("")).alias("user_id"),
        F.coalesce(F.col("time"), F.lit("")).alias("event_time"),
        F.coalesce(F.col("history"), F.lit("")).alias("history"),
        F.coalesce(F.col("article_ref"), F.lit("")).alias("article_id"),
        F.coalesce(F.col("category"), F.lit("general")).alias("category"),
        F.coalesce(F.col("clicked"), F.lit(0)).alias("clicked")
    )
)

# write processed behaviors parquet
behaviors_clean.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/behaviors/")


# -----------------------------
# 6) BUILD USER VECTORS
# -----------------------------

print("Building user vectors...")

# simulated clicks only
sim_clicks = (
    logs_clean
    .filter(F.col("action") == "click")
    .select(
        "user_id",
        "article_id",
        "category",
        F.col("timestamp").alias("event_time")
    )
)

# MIND clicked impressions only
mind_clicks = (
    behaviors_clean
    .filter(F.col("clicked") == 1)
    .select(
        "user_id",
        "article_id",
        "category",
        "event_time"
    )
)

# combine both click sources
all_clicks = sim_clicks.unionByName(mind_clicks)

# keep only your fixed presentation categories for vector counts
clicks_for_counts = (
    all_clicks
    .withColumn("category", F.lower(F.col("category")))
    .filter(F.col("category").isin(FIXED_CATEGORIES))
)

# pivot to one row per user with category counts
user_category_counts = (
    clicks_for_counts
    .groupBy("user_id")
    .pivot("category", FIXED_CATEGORIES)
    .count()
    .fillna(0)
)

# rank clicks per user by most recent event_time string
recent_window = Window.partitionBy("user_id").orderBy(F.col("event_time").desc_nulls_last())

# keep up to 10 recent clicked article ids
recent_clicks = (
    all_clicks
    .withColumn("rn", F.row_number().over(recent_window))
    .filter(F.col("rn") <= 10)
    .groupBy("user_id")
    .agg(F.collect_list("article_id").alias("recent_clicks"))
)

# merge counts and recent clicks
user_vectors = user_category_counts.join(recent_clicks, on="user_id", how="full_outer")

# fill missing category counts with 0
for cat in FIXED_CATEGORIES:
    if cat in user_vectors.columns:
        user_vectors = user_vectors.withColumn(cat, F.coalesce(F.col(cat), F.lit(0)))
    else:
        user_vectors = user_vectors.withColumn(cat, F.lit(0))

# fill missing recent_clicks with empty array
user_vectors = user_vectors.withColumn(
    "recent_clicks",
    F.when(F.col("recent_clicks").isNull(), F.array().cast("array<string>")).otherwise(F.col("recent_clicks"))
)

# total clicks across the fixed categories
user_vectors = user_vectors.withColumn(
    "total_clicks",
    F.col("business") +
    F.col("entertainment") +
    F.col("technology") +
    F.col("health") +
    F.col("science")
)

# choose top category; if no clicks then mark unknown
user_vectors = user_vectors.withColumn(
    "top_category",
    F.when(
        F.col("total_clicks") == 0,
        F.lit("unknown")
    ).otherwise(
        F.expr("""
            CASE
                WHEN business >= entertainment AND business >= technology AND business >= health AND business >= science THEN 'business'
                WHEN entertainment >= technology AND entertainment >= health AND entertainment >= science THEN 'entertainment'
                WHEN technology >= health AND technology >= science THEN 'technology'
                WHEN health >= science THEN 'health'
                ELSE 'science'
            END
        """)
    )
)

# write processed user vectors parquet
user_vectors.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/vectors/")

print("Expanded ETL finished successfully.")
job.commit()