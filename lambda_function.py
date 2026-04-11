# This code goes inside the AWS Lambda Function code editor, not in VS Code terminal.

import json                                                              # Import json so the Lambda function can parse API responses and save article data as JSON.
import os                                                                # Import os so the function can read the NEWSAPI_KEY from Lambda Environment Variables.
import boto3                                                             # Import boto3 so the function can upload the fetched article file to S3.
import urllib.request                                                    # Import urllib.request so the function can send HTTP requests to NewsAPI.
import urllib.parse                                                      # Import urllib.parse so the API query parameters can be encoded safely into the URL.
from datetime import datetime, timezone                                  # Import datetime tools so the function can generate UTC timestamps for each run.

S3_BUCKET = "news-recommending-raw-st125934"                             # Name of the raw S3 bucket where each NewsAPI result file will be saved.
NEWSAPI_KEY = os.environ["NEWSAPI_KEY"]                                  # Read the NewsAPI key from Lambda Environment Variables instead of hardcoding it in the script.
CATEGORIES = ["business", "entertainment", "technology", "health", "science"]  # List of categories the Lambda function will fetch from NewsAPI every run.


def lambda_handler(event, context):                                      # Define the main Lambda handler function that AWS will call when the trigger runs.
    s3 = boto3.client("s3")                                              # Create an S3 client so the function can upload the fetched JSON file to your bucket.
    all_articles = []                                                    # Create an empty list to collect all articles from all categories in one run.

    for category in CATEGORIES:                                          # Loop through each category in the predefined category list.
        print(f"Fetching category: {category}")                          # Print the current category name to see progress in CloudWatch logs.

        params = urllib.parse.urlencode({                                # Build the NewsAPI query string safely using encoded URL parameters.
            "apiKey": NEWSAPI_KEY,                                       # Pass the NewsAPI key so the request is authorized.
            "category": category,                                        # Request only the current category in this loop.
            "country": "us",                                             # Limit the API result to US headlines in this example.
            "pageSize": 20                                               # Request up to 20 articles per category.
        })

        url = f"https://newsapi.org/v2/top-headlines?{params}"           # Combine the base NewsAPI endpoint with the encoded query string.

        try:                                                             # Start a try block so one failed category does not crash the whole Lambda run.
            req = urllib.request.Request(url)                            # Build the HTTP request object for the NewsAPI call.
            with urllib.request.urlopen(req) as response:                # Send the request and open the HTTP response safely.
                data = json.loads(response.read().decode())              # Read the response body, decode it, and convert it from JSON text into a Python dictionary.
                articles = data.get("articles", [])                      # Extract the articles list from the API response, or use an empty list if missing.

                for article in articles:                                 # Loop through every article returned for the current category.
                    article["category"] = category                       # Add the current category into each article record so later ETL can use it directly.
                    article["fetched_at"] = datetime.now(timezone.utc).isoformat()  # Add the exact UTC fetch timestamp to each article row.

                all_articles.extend(articles)                            # Add all articles from this category into the combined all_articles list.
                print(f"Fetched {len(articles)} articles for {category}")# Print how many articles were fetched for this category.

        except Exception as e:                                           # Catch any error during API request or parsing.
            print(f"Error fetching {category}: {str(e)}")                # Print the error in CloudWatch logs but continue with the next category.

    print(f"Total fetched articles: {len(all_articles)}")                # Print the total number of articles collected across all categories.

    if all_articles:                                                     # Check whether at least one article was fetched before saving to S3.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") # Create a timestamp string so each output JSON file has a unique name.
        key = f"newsapi/{timestamp}_articles.json"                       # Build the final S3 object path inside the newsapi/ folder.

        s3.put_object(                                                   # Upload the combined article list as one JSON file to S3.
            Bucket=S3_BUCKET,                                            # Specify the destination S3 bucket.
            Key=key,                                                     # Specify the destination S3 folder path and file name.
            Body=json.dumps(all_articles, ensure_ascii=False, indent=2), # Convert the article list into nicely formatted JSON text before uploading.
            ContentType="application/json"                               # Mark the uploaded S3 object as a JSON file.
        )

        print(f"Saved {len(all_articles)} articles to s3://{S3_BUCKET}/{key}")  # Print the final S3 path so to verify it in CloudWatch logs.

    return {                                                             # Return a standard Lambda response object after the function finishes.
        "statusCode": 200,                                               # Return HTTP-style status code 200 to show the run succeeded.
        "body": json.dumps({                                             # Convert the response body dictionary into JSON text.
            "articles_fetched": len(all_articles),                       # Report how many total articles were fetched in this Lambda run.
            "timestamp": datetime.now(timezone.utc).isoformat()          # Report the UTC timestamp of the completed run.
        })
    }
