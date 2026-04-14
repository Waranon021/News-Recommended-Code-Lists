import requests                                                         # Import requests so the script can send an HTTP POST request to the live API.
import json                                                             # Import json to maybe print the returned response later.

API_URL = "https://ugdabibrak.execute-api.ap-southeast-7.amazonaws.com/prod/recommend"  # Final API Gateway endpoint for the deployed recommendation API.

payload = {                                                             # Build the request body that will be sent to the recommendation API.
    "user_id": "U8355",                                                 # Use one known user_id that already worked in Lambda testing.
    "k": 5                                                              # Ask the API to return 5 recommendations.
}

response = requests.post(                                               # Send a POST request to the live API Gateway endpoint.
    API_URL,                                                            # Use the deployed API Gateway URL.
    json=payload,                                                       # Send the request body as JSON.
    headers={"Content-Type": "application/json"}                        # Tell the API that this request body is JSON.
)

print(f"Status: {response.status_code}")                                # Print the HTTP status code returned by the API.
result = response.json()                                                # Convert the API JSON response into a normal Python dictionary.

print(f"Model used: {result.get('model_type')}")                        # Print the model type used by the serving Lambda.
print(f"Total recommendations: {result.get('total')}")                  # Print how many recommendations were returned.
print("\nRecommendations:")                                             # Print a heading before listing the returned articles.

for i, article in enumerate(result.get("recommendations", []), 1):      # Loop through each returned article and number them starting from 1.
    print(f"{i}. [{article.get('category', '')}] {article.get('title', '')}")  # Print category and title for each recommendation.
    print(f"   Source: {article.get('source', '')}")                    # Print the article source under the title.
    print(f"   URL: {article.get('url', '')}")                          # Print the article URL under the source.
    print()                                                             # Print a blank line so each recommendation is easier to read.