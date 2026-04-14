import requests                                                                 # Import requests so the script can call the deployed API Gateway endpoint.
import sys                                                                      # Import sys so the script can stop with an error code if the test fails.

API_URL = "https://ugdabibrak.execute-api.ap-southeast-7.amazonaws.com/prod/recommend"  # Final deployed recommendation API endpoint.

payload = {                                                                     # Build a realistic API request payload.
    "user_id": "U8355",                                                         # Use a known user_id that worked in earlier testing.
    "k": 5                                                                      # Ask the API for 5 recommendations.
}

response = requests.post(                                                       # Send the POST request to the deployed API endpoint.
    API_URL,                                                                    # Use the API Gateway URL.
    json=payload,                                                               # Send the request body as JSON.
    headers={"Content-Type": "application/json"}                                # Tell the API the body is JSON.
)

print(f"HTTP status: {response.status_code}")                                   # Print the returned HTTP status code.
if response.status_code != 200:                                                 # Check whether the API failed.
    print("E2E test failed: API did not return 200.")                           # Print an error if the API did not succeed.
    sys.exit(1)                                                                 # Exit with failure code.

result = response.json()                                                        # Parse the returned JSON body.
print(result)                                                                   # Print the full result to inspect the response.

if "recommendations" not in result:                                             # Check whether the recommendations field is missing.
    print("E2E test failed: recommendations field missing.")                    # Print an error message if response shape is wrong.
    sys.exit(1)                                                                 # Exit with failure code.

if len(result.get("recommendations", [])) == 0:                                 # Check whether no recommendation items were returned.
    print("E2E test failed: no recommendations returned.")                      # Print an error message if the recommendation list is empty.
    sys.exit(1)                                                                 # Exit with failure code.

print("E2E test passed successfully.")                                          # Print final success message if the whole API worked correctly.