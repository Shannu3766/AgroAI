import time
import requests

STATUS_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/status"
RELOAD_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/reload"

def check_status():
    try:
        response = requests.get(STATUS_URL)
        response.raise_for_status()
        return response.json().get("status")
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

def reload_model():
    try:
        response = requests.post(RELOAD_URL, headers={"Content-Type": "application/json"}, json={})
        print("Reload triggered:", response.status_code, response.json())
    except Exception as e:
        print(f"Error reloading model: {e}")

def monitor():
    while True:
        status = check_status()
        print("Current status:", status)

        if status == "unloaded":
            print("Model not loaded. Triggering reload...")
            reload_model()
            time.sleep(30)
        elif status == "ready":
            print("Model is ready.")

        time.sleep(10)

if __name__ == "__main__":
    monitor()
