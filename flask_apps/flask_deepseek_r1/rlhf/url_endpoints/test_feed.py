import requests

url = "https://deepseek-test-742894389221.us-central1.run.app/feedback"
feedback = {
    "prompt": "Given the following soil and environmental parameters:\n"
              "- Temperature: 30°C\n"
              "- Humidity: 60%\n"
              "- Moisture: 25%\n"
              "- Soil Type: Loamy\n"
              "- Nitrogen: 40 ppm\n"
              "- Potassium: 20 ppm\n"
              "- Phosphorous: 15 ppm\n\n"
              "Predict the suitable Crop Type and Fertilizer Name",
    "chosen": "Crop: Wheat\nFertilizer: Urea",
    "rejected": "Crop: Cotton\nFertilizer: DAP"
}

response = requests.post(url, json=feedback)

print("Status:", response.status_code)
# print("Response:", response.json())
