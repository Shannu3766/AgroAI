import requests

# Replace this with your actual deployed Cloud Run endpoint
PREDICT_URL = "https://deepseek-flask-gpu-service1-742894389221.us-central1.run.app/predict"

# Example input data matching the required parameters
payload = {
    "Temparature": 30.0,
    "Humidity": 45.0,
    "Moisture": 12.0,
    "Soil Type": "red",
    "Nitrogen": 45,
    "Potassium": 3,
    "Phosphorous": 34
}

try:
    response = requests.post(PREDICT_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)


# PS C:\Users\sivaj> Invoke-RestMethod -Method POST `
# >>   -Uri "https://deepseek-flask-gpu-service1-742894389221.us-central1.run.app/predict" `
# >>   -ContentType "application/json" `
# >>   -Body (@{
# >>     "Temparature" = 30.0
# >>     "Humidity" = 45.0
# >>     "Moisture" = 12.0
# >>     "Soil Type" = "red"
# >>     "Nitrogen" = 45
# >>     "Potassium" = 3
# >>     "Phosphorous" = 34
# >>   } | ConvertTo-Json -Depth 10)