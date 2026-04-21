import json                                                    # Import json to parse API Gateway request bodies and return JSON responses.
import boto3                                                   # Import boto3 to connect this Lambda function to AWS services such as DynamoDB.
from datetime import datetime, timezone                        # Import datetime and timezone to store the click timestamp in UTC.
from decimal import Decimal                                    # Import Decimal because DynamoDB stores numeric values using Decimal instead of float.

REGION = "ap-southeast-7"                                      # Set the AWS region where DynamoDB tables are deployed.
USERS_TABLE = "user-vectors"                                   # Set the DynamoDB table name that stores user profiles and click history.
ARTICLES_TABLE = "article-embeddings"                          # Set the DynamoDB table name that stores article metadata such as category.

dynamodb = boto3.resource("dynamodb", region_name=REGION)      # Create one DynamoDB resource object connected to the chosen AWS region.
users_table = dynamodb.Table(USERS_TABLE)                      # Create a table object for the user-vectors table so it can read and update user data.
articles_table = dynamodb.Table(ARTICLES_TABLE)                # Create a table object for the article-embeddings table so it can read clicked article details.


def decimal_default(obj):                                      # Define a helper function to convert DynamoDB Decimal values before JSON output.
    if isinstance(obj, Decimal):                               # Check whether the current object is a Decimal value from DynamoDB.
        return float(obj)                                      # Convert Decimal to float so json.dumps() can serialize it without error.
    raise TypeError                                            # Raise an error if the object type is unsupported by this helper.


def lambda_handler(event, context):                            # Define the main Lambda handler that AWS runs whenever the /click endpoint is called.
    try:                                                       # Start a try block so unexpected errors can be caught and returned cleanly.
        body = event.get("body", event)                        # Get the request body from API Gateway, or use the whole event directly in manual tests.
        if isinstance(body, str):                              # Check whether the body is a JSON string, which is common with Lambda proxy integration.
            body = json.loads(body)                            # Convert the JSON string body into a Python dictionary.

        user_id = str(body["user_id"])                         # Read the user_id from the request body and force it to string format.
        article_id = str(body["article_id"])                   # Read the article_id from the request body and force it to string format.

        article_resp = articles_table.get_item(Key={"article_id": article_id})   # Fetch the clicked article from DynamoDB using article_id as the primary key.
        article = article_resp.get("Item")                     # Extract the actual article record from the DynamoDB response.

        if not article:                                        # Check whether the article was not found in the article-embeddings table.
            return {                                           # Return an HTTP-style response dictionary back to API Gateway.
                "statusCode": 404,                             # Use 404 to indicate that the requested article does not exist.
                "headers": {                                   # Set response headers for the returned JSON message.
                    "Content-Type": "application/json",        # Tell the client that the response body is JSON.
                    "Access-Control-Allow-Origin": "*"         # Allow cross-origin requests so Streamlit or browser apps can call this endpoint.
                },
                "body": json.dumps({"error": f"Article not found: {article_id}"})  # Return a helpful error message showing which article_id was missing.
            }

        category = article.get("category", "unknown")          # Read the article category from the article record, or default to "unknown" if missing.

        user_resp = users_table.get_item(Key={"user_id": user_id})   # Fetch the current user profile from DynamoDB using user_id as the primary key.
        user = user_resp.get("Item")                           # Extract the existing user profile record if one is found.

        if user:                                               # Check whether this user already exists in the user-vectors table.
            recent_clicks = [str(x) for x in user.get("recent_clicks", [])]   # Read the existing recent_clicks list and convert every item to string.

            recent_clicks = [x for x in recent_clicks if x != article_id]      # Remove the clicked article if it is already in the list to avoid duplicates.
            recent_clicks.insert(0, article_id)                # Insert the newly clicked article at the front so it becomes the most recent click.
            recent_clicks = recent_clicks[:20]                 # Keep only the latest 20 clicked articles to stop the history from growing too large.

            category_prefs = user.get("category_preferences", {})              # Read the existing category preference dictionary from the user profile.
            current_count = Decimal(str(category_prefs.get(category, 0)))      # Read the current count for this category and convert it safely to Decimal.
            category_prefs[category] = current_count + Decimal("1")            # Increase the clicked category count by 1.

            top_category = max(category_prefs, key=lambda k: float(category_prefs[k]))   # Find the category with the highest click count after the update.
            total_clicks = Decimal(str(user.get("total_clicks", 0))) + Decimal("1")      # Increase the total number of clicks recorded for this user.

            users_table.update_item(                           # Update the existing user profile in DynamoDB with the new click information.
                Key={"user_id": user_id},                     # Identify which user record should be updated.
                UpdateExpression="""                           # Define which fields should be replaced in the user profile.
                    SET recent_clicks = :rc,                   # Replace recent_clicks with the updated click history list.
                        category_preferences = :prefs,         # Replace category_preferences with the updated category count dictionary.
                        top_category = :top,                   # Replace top_category with the currently strongest preference.
                        total_clicks = :tc,                    # Replace total_clicks with the new total number of recorded clicks.
                        last_updated = :ts                     # Replace last_updated with the current UTC timestamp.
                """,
                ExpressionAttributeValues={                    # Provide the actual values used by the placeholders in UpdateExpression.
                    ":rc": recent_clicks,                     # Pass the updated recent click list.
                    ":prefs": category_prefs,                 # Pass the updated category preference dictionary.
                    ":top": top_category,                     # Pass the updated top category.
                    ":tc": total_clicks,                      # Pass the updated total click count.
                    ":ts": datetime.now(timezone.utc).isoformat()  # Pass the current UTC timestamp in ISO format.
                }
            )

        else:                                                  # Run this block if the user profile does not yet exist in DynamoDB.
            users_table.put_item(                              # Create a brand-new user profile record in the user-vectors table.
                Item={                                         # Define all fields that should be stored for the new user.
                    "user_id": user_id,                        # Store the new user's ID.
                    "recent_clicks": [article_id],            # Start the click history list with the current clicked article.
                    "category_preferences": {                 # Start the category preference dictionary for the clicked category.
                        category: Decimal("1")                # Set the clicked category count to 1 for this new user.
                    },
                    "top_category": category,                 # Set the current top category to the clicked article's category.
                    "total_clicks": Decimal("1"),            # Set the total click count to 1 for this first recorded click.
                    "last_updated": datetime.now(timezone.utc).isoformat()  # Store the current UTC timestamp for audit and tracking purposes.
                }
            )
            top_category = category                            # Set top_category locally so it can also be returned in the response body.

        return {                                               # Return a success response if the click was recorded correctly.
            "statusCode": 200,                                 # Use 200 to show that the request succeeded.
            "headers": {                                       # Define response headers for the success response.
                "Content-Type": "application/json",            # Tell the client that the response is JSON.
                "Access-Control-Allow-Origin": "*"             # Allow browser-based applications from any origin to receive the response.
            },
            "body": json.dumps(                                # Convert the response dictionary into a JSON string for API Gateway.
                {
                    "user_id": user_id,                        # Return the user_id that was updated.
                    "article_id": article_id,                  # Return the article_id that was clicked.
                    "category": category,                      # Return the category of the clicked article.
                    "top_category": top_category,              # Return the user's updated top category after this click.
                    "message": "Click recorded successfully"   # Return a simple success message for the frontend.
                },
                default=decimal_default                        # Use the helper function so Decimal values can be safely serialized.
            )
        }

    except Exception as e:                                     # Catch any unexpected error that happens anywhere in the function.
        import traceback                                       # Import traceback only when needed so it can print the full error stack.
        traceback.print_exc()                                  # Print the full traceback to CloudWatch Logs for debugging.
        return {                                               # Return a failure response instead of crashing silently.
            "statusCode": 500,                                 # Use 500 to indicate an internal server error.
            "headers": {                                       # Define response headers for the error response.
                "Content-Type": "application/json",            # Tell the client that the error response is JSON.
                "Access-Control-Allow-Origin": "*"             # Allow the frontend to receive the error response across origins.
            },
            "body": json.dumps({"error": str(e)})              # Return the exception message so the frontend or tester can see the cause.
        }