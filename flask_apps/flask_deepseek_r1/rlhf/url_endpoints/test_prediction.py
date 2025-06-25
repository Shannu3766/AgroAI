import requests
import json

url = "https://deepseek-test-742894389221.us-central1.run.app/predict"

data = {
    "Temparature": 26,
    "Humidity": 52,
    "Moisture": 38,
    "Soil Type": "Loamy",
    "Nitrogen": 37,
    "Potassium": 0,
    "Phosphorous": 0
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)

    full_result = response.json()

    # Save the full response to a JSON file
    with open("recommendation_full.json", "w") as f:
        json.dump(full_result, f, indent=4)

    recommendation = full_result.get("recommendation", {})

    crop = recommendation.get("Recommended Crop Type", "Not found")
    fertilizer = recommendation.get("Recommended Fertilizer", "Not found")

    print("Crop Type:", crop)
    print("Fertilizer Name:", fertilizer)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except ValueError:
    print("Response is not valid JSON")
