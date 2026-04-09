import json
import random
import boto3
from datetime import datetime, timedelta, timezone

S3_BUCKET = "news-recommender-raw-st125934"
NUM_USERS = 100
NUM_INTERACTIONS = 1000
CATEGORIES = ["business", "entertainment", "technology", "health", "science"]

random.seed(42)

fake_article_ids = [f"article_{i:04d}" for i in range(200)]
interactions = []
base_time = datetime.now(timezone.utc)

for i in range(NUM_INTERACTIONS):
    user_id = f"user_{random.randint(1, NUM_USERS):03d}"
    article_id = random.choice(fake_article_ids)
    category = random.choice(CATEGORIES)
    action = random.choices(["click", "view", "skip"], weights=[0.3, 0.5, 0.2])[0]
    timestamp = base_time - timedelta(minutes=random.randint(0, 60 * 24 * 30))

    interactions.append({
        "user_id": user_id,
        "article_id": article_id,
        "category": category,
        "action": action,
        "timestamp": timestamp.isoformat(),
        "session_id": f"session_{random.randint(1000, 9999)}"
    })

interactions.sort(key=lambda x: x["timestamp"])

with open("simulated_user_logs.json", "w", encoding="utf-8") as f:
    json.dump(interactions, f, indent=2)

s3 = boto3.client("s3", region_name="us-east-1")
with open("simulated_user_logs.json", "rb") as f:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="user-logs/simulated_user_logs.json",
        Body=f,
        ContentType="application/json"
    )

print("Uploaded simulated logs to S3")
print("Local file saved as simulated_user_logs.json")