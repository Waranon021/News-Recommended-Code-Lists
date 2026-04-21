import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, to_json

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

RAW_BUCKET = "s3://news-recommender-raw-st125934/"
PROCESSED_BUCKET = "s3://news-recommender-processed-st125934/"

# 1. Read NewsAPI raw JSON
newsapi_path = RAW_BUCKET + "newsapi/"
news_df = spark.read.option("multiline", "true").json(newsapi_path)

if "source" in news_df.columns:
    news_df = news_df.withColumn("source_json", to_json(col("source"))).drop("source")

# Write processed news articles as Parquet
news_df.write.mode("overwrite").parquet(PROCESSED_BUCKET + "articles/")

# 2. Read simulated user logs
logs_path = RAW_BUCKET + "user-logs/"
logs_df = spark.read.option("multiline", "true").json(logs_path)

# Write processed user logs as Parquet
logs_df.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/logs/")

job.commit()