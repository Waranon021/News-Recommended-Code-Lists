import sys                                                                                     # Import sys so Glue can read runtime arguments like the job name.

from awsglue.utils import getResolvedOptions                                                   # Import Glue helper for reading job arguments such as JOB_NAME.
from pyspark.context import SparkContext                                                       # Import SparkContext to start the Spark environment inside Glue.
from awsglue.context import GlueContext                                                        # Import GlueContext to access Glue + Spark features together.
from awsglue.job import Job                                                                    # Import Job so the Glue script can start and commit the job properly.

from pyspark.sql import functions as F                                                         # Import Spark SQL functions with alias F for easier dataframe transformations.
from pyspark.sql.window import Window                                                          # Import Window so the script can rank and deduplicate rows with row_number.


args = getResolvedOptions(sys.argv, ["JOB_NAME"])                                              # Read the Glue job name from runtime arguments.

sc = SparkContext()                                                                            # Start the Spark context for distributed processing.
glueContext = GlueContext(sc)                                                                  # Wrap the Spark context inside a Glue context.
spark = glueContext.spark_session                                                              # Create the Spark session object used for reading and writing dataframes.
job = Job(glueContext)                                                                         # Create a Glue job object linked to this Glue context.
job.init(args["JOB_NAME"], args)                                                               # Start the Glue job using the received job name.

RAW_BUCKET = "s3://news-recommending-raw-st125934/"                                            # Define the raw S3 bucket path where NewsAPI JSON and MIND files are stored.
PROCESSED_BUCKET = "s3://news-recommending-processed-st125934/"                                # Define the processed S3 bucket path where cleaned parquet files will be written.

FIXED_CATEGORIES = ["business", "entertainment", "technology", "health", "science"]           # Define the fixed set of categories used later for user vector counts.


print("Loading NewsAPI JSON...")                                                               # Print a message to know the script is starting the NewsAPI load step.

newsapi_df = spark.read.option("multiline", "true").json(RAW_BUCKET + "newsapi/")             # Read all NewsAPI JSON files from raw/newsapi/ into one Spark dataframe.

if "source" in newsapi_df.columns:                                                             # Check whether the nested source object exists in the NewsAPI schema.
    newsapi_df = newsapi_df.withColumn("source_name", F.col("source.name")).drop("source")    # Flatten source.name into a normal column and remove the original nested source column.
else:                                                                                          # Handle the case where source does not exist.
    newsapi_df = newsapi_df.withColumn("source_name", F.lit("unknown"))                        # Create a fallback source_name column with the value unknown.

newsapi_std = newsapi_df.select(                                                               # Select and standardize the NewsAPI columns into a clean article schema.
    F.sha2(                                                                                    # Start building a stable hashed article_id.
        F.lower(                                                                               # Lowercase the chosen text before hashing so IDs stay consistent.
            F.coalesce(                                                                        # Use the first non-null value between URL and fallback text.
                F.trim(F.col("url")),                                                          # Prefer the cleaned URL as the main article identifier.
                F.concat_ws(                                                                   # If URL is missing, build fallback text from title + publishedAt + source_name.
                    "||",                                                                      # Join fallback fields with || so the combined text stays consistent.
                    F.coalesce(F.col("title"), F.lit("")),                                     # Use title as part of the fallback identifier.
                    F.coalesce(F.col("publishedAt"), F.lit("")),                               # Use publishedAt as part of the fallback identifier.
                    F.coalesce(F.col("source_name"), F.lit(""))                                # Use source_name as part of the fallback identifier.
                )
            )
        ),
        256                                                                                    # Use SHA-256 hashing length to build the final article_id.
    ).alias("article_id"),                                                                     # Name the hashed result article_id.
    F.coalesce(F.trim(F.col("url")), F.lit("")).alias("url"),                                 # Keep URL as a clean string and use empty string if it is missing.
    F.coalesce(F.col("title"), F.lit("")).alias("title"),                                     # Keep title and replace null with empty string.
    F.coalesce(F.col("description"), F.lit("")).alias("abstract"),                            # Map description into abstract and replace null with empty string.
    F.coalesce(F.col("content"), F.lit("")).alias("content"),                                 # Keep content and replace null with empty string.
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),               # Keep category, replace null with general, and lowercase it.
    F.coalesce(F.col("source_name"), F.lit("unknown")).alias("source"),                       # Save the flattened source_name as source with unknown fallback.
    F.coalesce(F.col("publishedAt"), F.lit("")).alias("pub_date"),                            # Save publishedAt into pub_date with empty fallback.
    F.coalesce(F.col("fetched_at"), F.lit("")).alias("fetched_at"),                           # Keep fetched_at and replace null with empty string.
    F.lit("newsapi").alias("data_source")                                                      # Mark these rows as coming from NewsAPI.
)

newsapi_std = newsapi_std.withColumn(                                                          # Add an article_key column used only for deduplication.
    "article_key",                                                                             # Name of the temporary deduplication key.
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))          # Use lowercase URL if present, otherwise fall back to article_id.
)

newsapi_window = Window.partitionBy("article_key").orderBy(                                    # Create a window to rank duplicate NewsAPI rows by most recent version.
    F.col("fetched_at").desc_nulls_last(),                                                     # Prefer the latest fetched_at value first.
    F.col("pub_date").desc_nulls_last()                                                        # If fetched_at ties, prefer the latest pub_date.
)

newsapi_dedup = (                                                                              # Start a deduplication pipeline for NewsAPI rows.
    newsapi_std                                                                                # Use the standardized NewsAPI dataframe as input.
    .withColumn("rn", F.row_number().over(newsapi_window))                                     # Add row numbers inside each duplicate group.
    .filter(F.col("rn") == 1)                                                                  # Keep only the top-ranked row from each duplicate group.
    .drop("rn", "article_key")                                                                 # Remove temporary ranking and dedup key columns after deduplication.
)

print("NewsAPI rows after dedup:")                                                             # Print a label before the NewsAPI row count.
print(newsapi_dedup.count())                                                                   # Print the final deduplicated NewsAPI row count.


print("Loading MIND news TSV...")                                                              # Print a message to know the script is starting the MIND news load step.

mind_news_raw = (                                                                              # Start a multi-line Spark read for the MIND news folder.
    spark.read                                                                                 # Use Spark read API to load raw files.
    .option("sep", "\t")                                                                       # Tell Spark the file is tab-separated.
    .option("header", "false")                                                                 # Tell Spark there is no header row in MIND news.tsv.
    .csv(RAW_BUCKET + "mind/news/")                                                            # Read all files inside raw/mind/news/.
)

mind_news_df = mind_news_raw.toDF(                                                             # Assign readable column names to the raw MIND news dataframe.
    "news_id",                                                                                 # Original MIND article ID.
    "category",                                                                                # MIND article category.
    "subcategory",                                                                             # MIND article subcategory.
    "title",                                                                                   # MIND article title.
    "abstract",                                                                                # MIND article abstract.
    "url",                                                                                     # MIND article URL.
    "title_entities",                                                                          # MIND title entity metadata.
    "abstract_entities"                                                                        # MIND abstract entity metadata.
)

mind_news_std = mind_news_df.select(                                                           # Standardize the MIND news dataframe into the same article schema as NewsAPI.
    F.sha2(                                                                                    # Start building hashed article_id for MIND news rows.
        F.lower(                                                                               # Lowercase the chosen identifier text before hashing.
            F.coalesce(                                                                        # Use the first non-null value between URL and news_id.
                F.trim(F.col("url")),                                                          # Prefer URL if present.
                F.trim(F.col("news_id"))                                                       # Otherwise use news_id as fallback identifier.
            )
        ),
        256                                                                                    # Use SHA-256 hashing length for stable IDs.
    ).alias("article_id"),                                                                     # Name the hashed result article_id.
    F.coalesce(                                                                                # Create the url field using real URL or fallback MIND-style URL.
        F.trim(F.col("url")),                                                                  # Prefer the real URL if it exists.
        F.concat(F.lit("mind://"), F.col("news_id"))                                           # Otherwise build a fake MIND URL using the news_id.
    ).alias("url"),                                                                            # Name the final field url.
    F.coalesce(F.col("title"), F.lit("")).alias("title"),                                     # Keep title with empty-string fallback.
    F.coalesce(F.col("abstract"), F.lit("")).alias("abstract"),                               # Keep abstract with empty-string fallback.
    F.lit("").alias("content"),                                                                # Set content as empty string because MIND news does not provide full content here.
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),               # Keep category, replace null with general, and lowercase it.
    F.lit("microsoft_mind").alias("source"),                                                   # Mark source as microsoft_mind.
    F.lit("").alias("pub_date"),                                                               # Leave pub_date empty because MIND news.tsv does not provide this field directly.
    F.lit("").alias("fetched_at"),                                                             # Leave fetched_at empty because these are static uploaded files.
    F.lit("mind").alias("data_source")                                                         # Mark these rows as coming from MIND.
)

mind_news_std = mind_news_std.withColumn(                                                      # Add an article_key column used only for MIND deduplication.
    "article_key",                                                                             # Name the temporary deduplication key.
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))          # Use lowercase URL if present, otherwise fall back to article_id.
)

mind_window = Window.partitionBy("article_key").orderBy(F.col("article_id"))                   # Create a window to rank duplicate MIND rows deterministically.

mind_news_dedup = (                                                                            # Start the MIND deduplication pipeline.
    mind_news_std                                                                              # Use the standardized MIND dataframe as input.
    .withColumn("rn", F.row_number().over(mind_window))                                        # Add row numbers inside each duplicate group.
    .filter(F.col("rn") == 1)                                                                  # Keep only the first row from each duplicate group.
    .drop("rn", "article_key")                                                                 # Remove temporary ranking and dedup key columns.
)

print("MIND news rows after dedup:")                                                           # Print a label before the MIND row count.
print(mind_news_dedup.count())                                                                 # Print the final deduplicated MIND article row count.


print("Combining NewsAPI + MIND articles...")                                                  # Print a message to know the script is now combining both article sources.

common_article_cols = [                                                                        # Define one shared ordered list of article columns for unioning both sources.
    "article_id",                                                                              # Stable article identifier.
    "url",                                                                                     # Article URL or fallback pseudo-URL.
    "title",                                                                                   # Article title.
    "abstract",                                                                                # Article abstract.
    "content",                                                                                 # Article content.
    "category",                                                                                # Article category.
    "source",                                                                                  # Original source name.
    "pub_date",                                                                                # Publication date.
    "fetched_at",                                                                              # Fetch time for dynamic NewsAPI rows.
    "data_source"                                                                              # Label showing which source the row came from.
]

articles_df = newsapi_dedup.select(common_article_cols).unionByName(                           # Union the cleaned NewsAPI articles with cleaned MIND articles.
    mind_news_dedup.select(common_article_cols)                                                # Use the same column order for MIND before unioning.
)

articles_df = articles_df.withColumn(                                                          # Add a final cross-source article_key for one more deduplication pass.
    "article_key",                                                                             # Name the temporary combined deduplication key.
    F.when(F.col("url") != "", F.lower(F.col("url"))).otherwise(F.col("article_id"))          # Use URL when present, otherwise use article_id.
)

combined_window = Window.partitionBy("article_key").orderBy(                                   # Create a ranking window across both sources.
    F.col("fetched_at").desc_nulls_last(),                                                     # Prefer the newest fetched_at first.
    F.col("pub_date").desc_nulls_last(),                                                       # Then prefer the newest pub_date.
    F.col("data_source").desc()                                                                # Then use data_source as a stable tie-breaker.
)

articles_df = (                                                                                # Start the final cross-source deduplication pipeline.
    articles_df                                                                                # Use the combined articles dataframe as input.
    .withColumn("rn", F.row_number().over(combined_window))                                    # Add ranking numbers inside each duplicate group.
    .filter(F.col("rn") == 1)                                                                  # Keep only the top-ranked row from each group.
    .drop("rn", "article_key")                                                                 # Remove temporary columns after deduplication.
)

print("Final combined article rows:")                                                          # Print a label before the final article count.
print(articles_df.count())                                                                     # Print the final combined deduplicated article row count.

articles_df.write.mode("overwrite").parquet(PROCESSED_BUCKET + "articles/")                    # Write the final article dataframe to processed/articles/ as parquet.


print("Loading simulated user logs JSON...")                                                   # Print a message to know the script is starting the user log load step.

logs_df = spark.read.option("multiline", "true").json(RAW_BUCKET + "user-logs/")              # Read the simulated user logs JSON files from raw/user-logs/.

logs_clean = logs_df.select(                                                                   # Select and clean the user log fields into a standard schema.
    F.coalesce(F.col("user_id"), F.lit("")).alias("user_id"),                                 # Keep user_id with empty-string fallback.
    F.coalesce(F.col("article_id"), F.lit("")).alias("article_id"),                           # Keep article_id with empty-string fallback.
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category"),               # Keep category, fallback to general, and lowercase it.
    F.lower(F.coalesce(F.col("action"), F.lit("view"))).alias("action"),                      # Keep action, fallback to view, and lowercase it.
    F.coalesce(F.col("session_id"), F.lit("")).alias("session_id"),                           # Keep session_id with empty-string fallback.
    F.coalesce(F.col("timestamp"), F.lit("")).alias("timestamp")                              # Keep timestamp with empty-string fallback.
)

logs_clean.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/logs/")                  # Write the cleaned user logs to processed/users/logs/ as parquet.


print("Loading MIND behaviors TSV...")                                                         # Print a message to know the script is starting the MIND behaviors load step.

mind_behaviors_raw = (                                                                         # Start a multi-line Spark read for the MIND behaviors folder.
    spark.read                                                                                 # Use Spark read API to load raw files.
    .option("sep", "\t")                                                                       # Tell Spark the file is tab-separated.
    .option("header", "false")                                                                 # Tell Spark there is no header row in MIND behaviors.tsv.
    .csv(RAW_BUCKET + "mind/behaviors/")                                                       # Read all files inside raw/mind/behaviors/.
)

mind_behaviors_df = mind_behaviors_raw.toDF(                                                   # Assign readable column names to the raw MIND behaviors dataframe.
    "impression_id",                                                                           # Impression identifier.
    "user_id",                                                                                 # User identifier.
    "time",                                                                                    # Event time string.
    "history",                                                                                 # Historical clicked items string.
    "impressions"                                                                              # Candidate impression list with click labels.
)

behaviors_exploded = (                                                                         # Start a transformation pipeline to explode the impressions list.
    mind_behaviors_df                                                                          # Use the raw behaviors dataframe as input.
    .withColumn("impression_item", F.explode(F.split(F.coalesce(F.col("impressions"), F.lit("")), " ")))  # Split the impressions string by spaces and explode into one row per impression item.
    .filter(F.trim(F.col("impression_item")) != "")                                            # Remove empty exploded items.
)

behaviors_parsed = (                                                                           # Start a parsing pipeline for each exploded impression item.
    behaviors_exploded                                                                         # Use the exploded behaviors dataframe as input.
    .withColumn("article_ref", F.regexp_extract(F.col("impression_item"), r"^(.*)-[01]$", 1))# Extract the raw MIND article reference such as N12345 from items like N12345-1.
    .withColumn("clicked", F.regexp_extract(F.col("impression_item"), r".*-(\d)$", 1).cast("int"))  # Extract the click label 0 or 1 and cast it to integer.
)

mind_news_lookup = mind_news_df.select(                                                        # Build a lookup table from MIND news_id to category.
    F.col("news_id").alias("article_ref"),                                                     # Rename news_id to article_ref so it matches the parsed behaviors column.
    F.lower(F.coalesce(F.col("category"), F.lit("general"))).alias("category")                # Keep category, fallback to general, and lowercase it.
).dropDuplicates(["article_ref"])                                                              # Keep one category row per MIND article_ref.

behaviors_clean = (                                                                            # Start the final behaviors cleaning pipeline.
    behaviors_parsed                                                                           # Use the parsed impression dataframe as input.
    .join(mind_news_lookup, on="article_ref", how="left")                                      # Join the parsed impression rows to the category lookup table.
    .select(                                                                                   # Select a clean standard schema for behaviors.
        F.coalesce(F.col("user_id"), F.lit("")).alias("user_id"),                              # Keep user_id with empty-string fallback.
        F.coalesce(F.col("time"), F.lit("")).alias("event_time"),                              # Save the original event time string as event_time.
        F.coalesce(F.col("history"), F.lit("")).alias("history"),                              # Keep history with empty-string fallback.
        F.coalesce(F.col("article_ref"), F.lit("")).alias("article_id"),                       # Rename article_ref into article_id for consistency with the training pipeline.
        F.coalesce(F.col("category"), F.lit("general")).alias("category"),                     # Keep category with general fallback.
        F.coalesce(F.col("clicked"), F.lit(0)).alias("clicked")                                # Keep clicked with 0 fallback.
    )
)

behaviors_clean.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/behaviors/")        # Write the cleaned behaviors dataframe to processed/users/behaviors/ as parquet.


print("Building user vectors...")                                                              # Print a message so to know the script is starting the user vector build step.

sim_clicks = (                                                                                 # Start a pipeline for simulated click interactions only.
    logs_clean                                                                                 # Use cleaned simulated logs as input.
    .filter(F.col("action") == "click")                                                        # Keep only rows where the action is click.
    .select(                                                                                   # Select only the fields needed for user vector construction.
        "user_id",                                                                             # Keep user_id.
        "article_id",                                                                          # Keep article_id.
        "category",                                                                            # Keep category.
        F.col("timestamp").alias("event_time")                                                 # Rename timestamp to event_time so it matches MIND clicks later.
    )
)

mind_clicks = (                                                                                # Start a pipeline for clicked MIND impressions only.
    behaviors_clean                                                                            # Use cleaned behaviors as input.
    .filter(F.col("clicked") == 1)                                                             # Keep only positive clicked impressions.
    .select(                                                                                   # Select only the fields needed for user vector construction.
        "user_id",                                                                             # Keep user_id.
        "article_id",                                                                          # Keep article_id.
        "category",                                                                            # Keep category.
        "event_time"                                                                           # Keep event_time.
    )
)

all_clicks = sim_clicks.unionByName(mind_clicks)                                               # Combine simulated clicks and MIND clicks into one unified click dataframe.

clicks_for_counts = (                                                                          # Start a pipeline used only for category count pivoting.
    all_clicks                                                                                 # Use all combined clicks as input.
    .withColumn("category", F.lower(F.col("category")))                                        # Lowercase category values for consistency.
    .filter(F.col("category").isin(FIXED_CATEGORIES))                                          # Keep only the fixed categories used.
)

user_category_counts = (                                                                       # Start a pipeline that creates one row per user with category click counts.
    clicks_for_counts                                                                          # Use cleaned clicks_for_counts as input.
    .groupBy("user_id")                                                                        # Group clicks by user_id.
    .pivot("category", FIXED_CATEGORIES)                                                       # Turn category values into separate pivoted count columns.
    .count()                                                                                   # Count clicks per user per category.
    .fillna(0)                                                                                 # Replace missing category counts with 0.
)

recent_window = Window.partitionBy("user_id").orderBy(F.col("event_time").desc_nulls_last())  # Create a window that ranks each user's clicks from most recent to oldest.

recent_clicks = (                                                                              # Start a pipeline for keeping each user's recent clicked article list.
    all_clicks                                                                                 # Use all combined clicks as input.
    .withColumn("rn", F.row_number().over(recent_window))                                      # Rank clicks for each user by most recent event_time first.
    .filter(F.col("rn") <= 10)                                                                 # Keep only the latest 10 clicks per user.
    .groupBy("user_id")                                                                        # Group the remaining clicks by user_id.
    .agg(F.collect_list("article_id").alias("recent_clicks"))                                 # Collect the article_id values into a list called recent_clicks.
)

user_vectors = user_category_counts.join(recent_clicks, on="user_id", how="full_outer")       # Join category counts and recent_clicks into one user vector dataframe.

for cat in FIXED_CATEGORIES:                                                                   # Loop through every fixed category to ensure all columns exist and have values.
    if cat in user_vectors.columns:                                                            # Check whether the current category column already exists.
        user_vectors = user_vectors.withColumn(cat, F.coalesce(F.col(cat), F.lit(0)))         # Replace null values in that existing category column with 0.
    else:                                                                                      # Handle the case where that category column does not exist at all.
        user_vectors = user_vectors.withColumn(cat, F.lit(0))                                  # Create the missing category column and fill it with 0.

user_vectors = user_vectors.withColumn(                                                        # Add or clean the recent_clicks column so null values become empty arrays.
    "recent_clicks",                                                                           # Name of the recent-click list column.
    F.when(F.col("recent_clicks").isNull(), F.array().cast("array<string>")).otherwise(F.col("recent_clicks"))  # Replace null recent_clicks with an empty string array.
)

user_vectors = user_vectors.withColumn(                                                        # Add a total_clicks column based on all fixed category counts.
    "total_clicks",                                                                            # Name of the total-click count column.
    F.col("business") +                                                                        # Add business click count.
    F.col("entertainment") +                                                                   # Add entertainment click count.
    F.col("technology") +                                                                      # Add technology click count.
    F.col("health") +                                                                          # Add health click count.
    F.col("science")                                                                           # Add science click count.
)

user_vectors = user_vectors.withColumn(                                                        # Add a top_category column that stores each user's strongest category.
    "top_category",                                                                            # Name of the top-category column.
    F.when(                                                                                    # Start a conditional expression for users with or without clicks.
        F.col("total_clicks") == 0,                                                            # Check whether the user has zero total clicks.
        F.lit("unknown")                                                                       # If zero clicks, label the top_category as unknown.
    ).otherwise(                                                                               # Otherwise calculate the top category using the category counts.
        F.expr("""                                                                             # Use a SQL CASE expression to compare the category columns.
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

user_vectors.write.mode("overwrite").parquet(PROCESSED_BUCKET + "users/vectors/")             # Write the final user vectors dataframe to processed/users/vectors/ as parquet.

print("Expanded ETL finished successfully.")                                                   # Print the final success message for the expanded ETL job.
job.commit()                                                                                   # Commit the Glue job so AWS marks it as completed successfully.
