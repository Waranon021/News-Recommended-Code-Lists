#This one put inside Lambda Function, not running it on VSCode
import json
import os
import boto3
import urllib.request
import urllib.parse
from datetime import datetime, timezone

S3_BUCKET = "news-recommender-raw-st125934"
NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]
CATEGORIES = ["business", "entertainment", "technology", "health", "science"]

def lambda_handler(event, context):
    s3 = boto3.client("s3")
    all_articles = []

    for category in CATEGORIES:
        print(f"Fetching category: {category}")

        params = urllib.parse.urlencode({
            "apiKey": NEWSAPI_KEY,
            "category": category,
            "country": "us",
            "pageSize": 20
        })

        url = f"https://newsapi.org/v2/top-headlines?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                articles = data.get("articles", [])

                for article in articles:
                    article["category"] = category
                    article["fetched_at"] = datetime.now(timezone.utc).isoformat()

                all_articles.extend(articles)
                print(f"Fetched {len(articles)} articles for {category}")

        except Exception as e:
            print(f"Error fetching {category}: {str(e)}")

    print(f"Total fetched articles: {len(all_articles)}")

    if all_articles:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        key = f"newsapi/{timestamp}_articles.json"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(all_articles, ensure_ascii=False, indent=2),
            ContentType="application/json"
        )

        print(f"Saved {len(all_articles)} articles to s3://{S3_BUCKET}/{key}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "articles_fetched": len(all_articles),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    }
