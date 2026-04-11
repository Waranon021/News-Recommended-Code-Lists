import json                                                             # Import json so the script can save the generated interactions into a local JSON file.
import random                                                           # Import random so the script can generate fake users, articles, actions, and timestamps.
import boto3                                                            # Import boto3 so the script can upload the generated JSON file to AWS S3.
from datetime import datetime, timedelta, timezone                      # Import datetime tools for creating realistic timestamps in UTC.

S3_BUCKET = "news-recommending-raw-st125934"                             # Name of the raw S3 bucket where the simulated user log file will be uploaded.
NUM_USERS = 100                                                         # Total number of fake users to simulate.
NUM_INTERACTIONS = 1000                                                 # Total number of simulated user interactions to generate.
CATEGORIES = ["business", "entertainment", "technology", "health", "science"]  # Categories used to label each fake interaction.

random.seed(42)                                                         # Set a fixed random seed so the generated data stays reproducible every time the script run.

fake_article_ids = [f"article_{i:04d}" for i in range(200)]             # Create 200 fake article IDs such as article_0000, article_0001, and so on.
interactions = []                                                       # Create an empty list to store all generated interaction records.
base_time = datetime.now(timezone.utc)                                  # Capture the current UTC time as the latest possible timestamp.

for i in range(NUM_INTERACTIONS):                                       # Loop 1000 times so the script creates 1000 interaction records.
    user_id = f"user_{random.randint(1, NUM_USERS):03d}"                # Randomly choose one fake user ID such as user_001 to user_100.
    article_id = random.choice(fake_article_ids)                        # Randomly choose one fake article ID from the article list.
    category = random.choice(CATEGORIES)                                # Randomly assign one category to this interaction.
    action = random.choices(["click", "view", "skip"], weights=[0.3, 0.5, 0.2])[0]  # Randomly choose an action, with view most likely, then click, then skip.
    timestamp = base_time - timedelta(minutes=random.randint(0, 60 * 24 * 30))       # Generate a timestamp sometime within the last 30 days.

    interactions.append({                                               # Add one complete interaction record into the interactions list.
        "user_id": user_id,                                             # Save the generated user ID.
        "article_id": article_id,                                       # Save the chosen fake article ID.
        "category": category,                                           # Save the chosen category.
        "action": action,                                               # Save the chosen action type.
        "timestamp": timestamp.isoformat(),                             # Convert the timestamp into ISO string format for JSON storage.
        "session_id": f"session_{random.randint(1000, 9999)}"           # Create a fake session ID to make the record look more realistic.
    })

interactions.sort(key=lambda x: x["timestamp"])                         # Sort all interactions by timestamp so the file looks chronologically ordered.

with open("simulated_user_logs.json", "w", encoding="utf-8") as f:      # Open a local file named simulated_user_logs.json for writing.
    json.dump(interactions, f, indent=2)                                # Save the interaction list into the local JSON file with nice indentation.

s3 = boto3.client("s3", region_name="ap-southeast-7")                   # Create an S3 client in the AWS region where your bucket is located.
with open("simulated_user_logs.json", "rb") as f:                       # Open the local JSON file again in binary read mode for uploading.
    s3.put_object(                                                      # Upload the local JSON file into the S3 bucket as a new object.
        Bucket=S3_BUCKET,                                               # Specify the destination S3 bucket name.
        Key="user-logs/simulated_user_logs.json",                       # Specify the S3 folder path and final uploaded file name.
        Body=f,                                                         # Use the local file content as the upload body.
        ContentType="application/json"                                  # Mark the uploaded object as a JSON file in S3.
    )

print("Uploaded simulated logs to S3")                                  # Print a success message after the upload is finished.
print("Local file saved as simulated_user_logs.json")                   # Print a success message for the local JSON file creation.
