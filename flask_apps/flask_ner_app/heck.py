import requests
import json

# Service URL
PREDICT_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/predict"

# Request data
data = {
    "Temparature": 30.0,
    "Humidity": 45.0,
    "Moisture": 12.0,
    "Soil Type": "red",
    "Nitrogen": 45,
    "Potassium": 3,
    "Phosphorous": 34
}

# Make the request
try:
    response = requests.post(
        PREDICT_URL,
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    # Check if request was successful
    response.raise_for_status()
    
    # Print the response
    print("\nResponse Status Code:", response.status_code)
    print("\nResponse Headers:", json.dumps(dict(response.headers), indent=2))
    print("\nResponse Body:", json.dumps(response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"\nError occurred: {str(e)}")
    if hasattr(e.response, 'text'):
        print(f"Response text: {e.response.text}")
