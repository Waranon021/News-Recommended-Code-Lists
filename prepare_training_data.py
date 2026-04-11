import os                                                                 # Import os so the script can build file paths safely on your computer.
import json                                                               # Import json so the script can save a summary file at the end.
import hashlib                                                            # Import hashlib to rebuild the same stable article_id logic used in ETL.
import pandas as pd                                                       # Import pandas for loading parquet files and preparing tabular training data.
from sklearn.model_selection import train_test_split                      # Import train_test_split to split the final interactions into train and test sets.


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                     # Get the folder where this current script is stored.
DATA_DIR = os.path.join(BASE_DIR, "training_data")                        # Point to the local folder created by download_training_data.py in Step 1.

ARTICLES_PATH = os.path.join(DATA_DIR, "articles.parquet")                # Local parquet file downloaded from processed/articles/.
BEHAVIORS_PATH = os.path.join(DATA_DIR, "behaviors.parquet")              # Local parquet file downloaded from processed/users/behaviors/.
USER_LOGS_PATH = os.path.join(DATA_DIR, "user_logs.parquet")              # Local parquet file downloaded from processed/users/logs/.
USER_VECTORS_PATH = os.path.join(DATA_DIR, "user_vectors.parquet")        # Local parquet file downloaded from processed/users/vectors/.

TRAIN_OUTPUT_PATH = os.path.join(DATA_DIR, "train_interactions.parquet")  # Output parquet file for training interactions used by train_models.py.
TEST_OUTPUT_PATH = os.path.join(DATA_DIR, "test_interactions.parquet")    # Output parquet file for testing interactions used by train_models.py.
ALL_OUTPUT_PATH = os.path.join(DATA_DIR, "all_interactions.parquet")      # Output parquet file for the full combined interactions dataset.
ARTICLES_OUTPUT_PATH = os.path.join(DATA_DIR, "training_articles.parquet")# Output parquet file for article metadata aligned to final interactions.
SUMMARY_OUTPUT_PATH = os.path.join(DATA_DIR, "training_summary.json")     # Output JSON file for a readable summary of the prepared dataset.

MIND_TRAIN_NEWS_PATH = os.path.join(BASE_DIR, "..", "MINDsmall_train", "news.tsv")  # Local raw MIND training news file used to map news_id to final article_id.
MIND_DEV_NEWS_PATH = os.path.join(BASE_DIR, "..", "MINDsmall_dev", "news.tsv")      # Local raw MIND dev news file used to map news_id to final article_id.

RANDOM_STATE = 42                                                         # Fixed random seed so train/test splitting stays reproducible every run.


def stable_article_id(url_value, fallback_value):                         # Define a helper function that rebuilds the same stable article_id logic from ETL.
    url_text = "" if pd.isna(url_value) else str(url_value).strip().lower()          # Use URL first because ETL preferred URL when it existed.
    fallback_text = "" if pd.isna(fallback_value) else str(fallback_value).strip().lower()  # Use fallback like news_id when URL is missing.
    source_text = url_text if url_text != "" else fallback_text           # Pick URL if available, otherwise use fallback text.

    if source_text == "":                                                 # Stop if both URL and fallback are empty.
        return None                                                       # Return None so bad rows can be dropped later.

    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()        # Hash the chosen text into a stable article_id string.


def normalize_time(series):                                               # Define a helper function for safely converting time columns into proper datetime values.
    return pd.to_datetime(series, errors="coerce", utc=True)              # Convert values to UTC datetime and turn bad values into NaT instead of crashing.


def load_mind_news_mapping():                                             # Define a helper function that reads local MIND news files and builds news_id -> final article_id mapping.
    mind_dfs = []                                                         # Create a list to hold train and dev MIND news tables.

    for path in [MIND_TRAIN_NEWS_PATH, MIND_DEV_NEWS_PATH]:               # Loop through both local MIND news.tsv files.
        if os.path.exists(path):                                          # Only read the file if it actually exists on your machine.
            df = pd.read_csv(                                             # Read the TSV file into pandas.
                path,                                                     # Use the current train or dev news file path.
                sep="\t",                                                 # Tell pandas that the file is tab-separated.
                header=None,                                              # MIND news.tsv does not come with a header row.
                names=[                                                   # Assign readable column names manually.
                    "news_id",                                            # Original MIND article ID like N12345.
                    "category",                                           # Article category in MIND.
                    "subcategory",                                        # Article subcategory in MIND.
                    "title",                                              # Article title in MIND.
                    "abstract",                                           # Article abstract in MIND.
                    "url",                                                # Article URL in MIND.
                    "title_entities",                                     # Extra entity info from MIND title.
                    "abstract_entities",                                  # Extra entity info from MIND abstract.
                ],
                usecols=[0, 5],                                           # Only load news_id and url because that is enough for ID mapping.
            )
            mind_dfs.append(df)                                           # Add the loaded dataframe to the list.

    if not mind_dfs:                                                      # Check whether no MIND news files were found.
        print("No local MIND news.tsv files found.")                      # Print a warning so you know the mapping step did not run.
        return {}                                                         # Return an empty dictionary so the script can continue safely.

    mind_news_df = pd.concat(mind_dfs, ignore_index=True)                 # Combine train and dev MIND news tables into one dataframe.
    mind_news_df = mind_news_df.drop_duplicates(subset=["news_id"])       # Keep one row per MIND news_id to avoid duplicate mappings.

    mind_news_df["article_id"] = mind_news_df.apply(                      # Create final article_id values for every MIND news row.
        lambda row: stable_article_id(row["url"], row["news_id"]),        # Apply the same hashing rule used in ETL.
        axis=1,                                                           # Run the lambda row by row.
    )

    mapping = dict(zip(mind_news_df["news_id"].astype(str), mind_news_df["article_id"]))  # Build a dictionary from raw MIND news_id to hashed article_id.

    print(f"MIND news_id mapping loaded: {len(mapping)} rows")            # Print how many mapping rows were created.
    return mapping                                                        # Return the dictionary so later code can map behaviors to final article IDs.


print("Loading local data.")                                              # Print a start message for the terminal.
articles_df = pd.read_parquet(ARTICLES_PATH)                              # Load local article metadata downloaded in Step 1.
behaviors_df = pd.read_parquet(BEHAVIORS_PATH)                            # Load local processed MIND behaviors downloaded in Step 1.
logs_df = pd.read_parquet(USER_LOGS_PATH)                                 # Load local processed simulated user logs downloaded in Step 1.
user_vectors_df = pd.read_parquet(USER_VECTORS_PATH)                      # Load local user vectors downloaded in Step 1.

print(f"Articles : {len(articles_df)}")                                   # Show article row count for quick verification.
print(f"Behaviors: {len(behaviors_df)}")                                  # Show behavior row count for quick verification.
print(f"Logs     : {len(logs_df)}")                                       # Show processed log row count for quick verification.
print(f"Users    : {len(user_vectors_df)}")                               # Show user vector row count for quick verification.

articles_df = articles_df.drop_duplicates(subset=["article_id"]).copy()   # Keep unique article_id rows because training should use one item definition per article.
valid_article_ids = set(articles_df["article_id"].astype(str).tolist())   # Build a set of valid article IDs for fast filtering later.

mind_mapping = load_mind_news_mapping()                                   # Build MIND news_id -> hashed article_id mapping from local raw news.tsv files.

print("=== Behaviors sample ===")                                         # Print a label before showing a small sample of the behaviors table.
print(behaviors_df.head(3))                                               # Print the first few behavior rows so you can visually inspect the data.

if "article_id" not in behaviors_df.columns:                              # Check that the processed behaviors parquet still has an article_id column.
    raise ValueError("behaviors.parquet does not contain article_id column.")  # Stop with a clear error if the column is missing.

print("Building interactions from MIND behaviors.")                       # Tell the terminal that the script is now extracting MIND click interactions.

mind_clicks_df = behaviors_df.copy()                                      # Make a working copy so the original dataframe is preserved.
mind_clicks_df["user_id"] = mind_clicks_df["user_id"].astype(str)         # Force user_id to string because user IDs should be handled as text.
mind_clicks_df["article_id_raw"] = mind_clicks_df["article_id"].astype(str)  # Keep the raw MIND news_id value before mapping it.
mind_clicks_df["clicked"] = pd.to_numeric(mind_clicks_df["clicked"], errors="coerce").fillna(0).astype(int)  # Convert clicked flags safely into integers.
mind_clicks_df = mind_clicks_df[mind_clicks_df["clicked"] == 1].copy()    # Keep only positive click events because these are the interactions used for recommendation training.
mind_clicks_df["article_id"] = mind_clicks_df["article_id_raw"].map(mind_mapping)  # Map raw MIND news_id values into final hashed article_id values.
mind_clicks_df["timestamp"] = normalize_time(mind_clicks_df["event_time"])  # Convert MIND event_time values into clean timestamps.

mind_clicks_df = mind_clicks_df[mind_clicks_df["article_id"].notna()].copy()  # Drop rows that could not be mapped to final article IDs.
mind_clicks_df["article_id"] = mind_clicks_df["article_id"].astype(str)   # Force final article_id into string type.
mind_clicks_df = mind_clicks_df[mind_clicks_df["article_id"].isin(valid_article_ids)].copy()  # Keep only clicks whose final article_id really exists in your article universe.

mind_clicks_df = mind_clicks_df[["user_id", "article_id", "timestamp"]].copy()  # Keep only the columns needed for model training.
mind_clicks_df["rating"] = 1.0                                           # Assign implicit-feedback value 1.0 to each click event.
mind_clicks_df["source"] = "mind_click"                                  # Label these rows as coming from MIND clicks.

print(f"MIND interactions extracted: {len(mind_clicks_df)}")             # Print how many valid MIND click interactions were created.

print("Adding simulated log interactions.")                              # Tell the terminal the script is now processing simulated user log clicks.

sim_clicks_df = logs_df.copy()                                           # Make a working copy of processed user logs.
sim_clicks_df["user_id"] = sim_clicks_df["user_id"].astype(str)          # Force user_id to string for consistency.
sim_clicks_df["action"] = sim_clicks_df["action"].astype(str).str.lower()  # Normalize action names like click/view/skip to lowercase.
sim_clicks_df = sim_clicks_df[sim_clicks_df["action"] == "click"].copy() # Keep only click events from simulated logs.
sim_clicks_df["article_id"] = sim_clicks_df["article_id"].astype(str)    # Force article_id to string.
sim_clicks_df["timestamp"] = normalize_time(sim_clicks_df["timestamp"])  # Convert simulated log timestamps into clean datetime values.

sim_clicks_df = sim_clicks_df[sim_clicks_df["article_id"].isin(valid_article_ids)].copy()  # Keep only simulated clicks that point to valid article IDs in your article dataset.

sim_clicks_df = sim_clicks_df[["user_id", "article_id", "timestamp"]].copy()  # Keep only training-relevant columns.
sim_clicks_df["rating"] = 1.0                                            # Assign implicit-feedback value 1.0 to each simulated click.
sim_clicks_df["source"] = "simulated_click"                              # Label these rows as simulated clicks.

all_interactions_df = pd.concat([mind_clicks_df, sim_clicks_df], ignore_index=True)  # Combine MIND clicks and simulated clicks into one training interaction table.
all_interactions_df = all_interactions_df.dropna(subset=["user_id", "article_id", "timestamp"]).copy()  # Remove any row missing key training fields.
all_interactions_df = all_interactions_df.drop_duplicates(subset=["user_id", "article_id", "timestamp", "source"]).copy()  # Remove exact duplicate interaction rows.

all_interactions_df["timestamp"] = normalize_time(all_interactions_df["timestamp"])  # Re-normalize timestamps after concatenation just to keep the final table consistent.
all_interactions_df = all_interactions_df.sort_values("timestamp").reset_index(drop=True)  # Sort interactions by time and reset the row index.

print("=== Interaction summary ===")                                     # Print a label for the interaction summary block.
print(f"Total interactions : {len(all_interactions_df)}")                # Print total number of final interactions.
print(f"Unique users       : {all_interactions_df['user_id'].nunique()}")  # Print number of distinct users in the final dataset.
print(f"Unique articles    : {all_interactions_df['article_id'].nunique()}")  # Print number of distinct articles in the final dataset.
print("By source:")                                                      # Print a label before source counts.
print(all_interactions_df["source"].value_counts())                      # Print counts for mind_click and simulated_click separately.

if len(all_interactions_df) == 0:                                        # Stop early if the final dataset ended up empty.
    raise ValueError("No interactions available after preparation.")      # Raise a clear error so you know training cannot continue.

train_df, test_df = train_test_split(                                    # Split the final interactions into train and test sets.
    all_interactions_df,                                                 # Use the full interaction dataframe as the input.
    test_size=0.20,                                                      # Keep 20% for testing and 80% for training.
    random_state=RANDOM_STATE,                                           # Use a fixed seed so the split is reproducible.
    shuffle=True,                                                        # Shuffle rows before splitting.
)

train_df = train_df.reset_index(drop=True)                               # Reset row numbering in the training dataframe.
test_df = test_df.reset_index(drop=True)                                 # Reset row numbering in the testing dataframe.

print(f"Train size: {len(train_df)}")                                    # Print how many rows are in the train set.
print(f"Test size : {len(test_df)}")                                     # Print how many rows are in the test set.

training_article_ids = set(all_interactions_df["article_id"].astype(str).tolist())  # Build a set of article IDs that actually appear in the final interactions.
training_articles_df = articles_df[articles_df["article_id"].astype(str).isin(training_article_ids)].copy()  # Keep only article metadata rows needed by the interaction dataset.

train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)                      # Save the train interactions parquet for train_models.py.
test_df.to_parquet(TEST_OUTPUT_PATH, index=False)                        # Save the test interactions parquet for evaluation in train_models.py.
all_interactions_df.to_parquet(ALL_OUTPUT_PATH, index=False)             # Save the full combined interactions parquet for backup or inspection.
training_articles_df.to_parquet(ARTICLES_OUTPUT_PATH, index=False)       # Save filtered article metadata aligned to final interactions.

summary = {                                                              # Create a summary dictionary for easy checking and note-taking.
    "articles_total": int(len(articles_df)),                             # Save total unique article rows loaded.
    "behaviors_total": int(len(behaviors_df)),                           # Save total processed behavior rows loaded.
    "logs_total": int(len(logs_df)),                                     # Save total processed user log rows loaded.
    "users_total": int(len(user_vectors_df)),                            # Save total user vector rows loaded.
    "mind_interactions": int(len(mind_clicks_df)),                       # Save count of valid MIND click interactions.
    "simulated_interactions": int(len(sim_clicks_df)),                   # Save count of valid simulated click interactions.
    "total_interactions": int(len(all_interactions_df)),                 # Save final interaction count after combining both sources.
    "unique_users": int(all_interactions_df["user_id"].nunique()),       # Save total unique users in final interactions.
    "unique_articles": int(all_interactions_df["article_id"].nunique()), # Save total unique articles in final interactions.
    "train_size": int(len(train_df)),                                    # Save train set size.
    "test_size": int(len(test_df)),                                      # Save test set size.
}

with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:              # Open a JSON file for writing the summary.
    json.dump(summary, f, indent=2)                                      # Save the summary dictionary as nicely formatted JSON.

print("Training data prepared and saved successfully.")                  # Print the final success message.
print(f"Sample MIND user IDs: {mind_clicks_df['user_id'].head(5).tolist()}")  # Print a few sample MIND user IDs so you can confirm real MIND users are present.