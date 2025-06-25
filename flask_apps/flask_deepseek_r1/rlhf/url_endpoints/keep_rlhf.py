import requests
import time

STATUS_ENDPOINT = "https://deepseek-test-742894389221.us-central1.run.app/status"
RELOAD_ENDPOINT = "https://deepseek-test-742894389221.us-central1.run.app/reload"

def get_status():
    try:
        res = requests.get(STATUS_ENDPOINT)
        if res.status_code == 200:
            status = res.json().get("status")
            print(f"[{time.ctime()}] Model status: {status}")
            return status
        else:
            print(f"[{time.ctime()}] Failed to fetch status. HTTP {res.status_code}")
            return None
    except Exception as e:
        print(f"[{time.ctime()}] Error during status check: {e}")
        return None

def trigger_reload():
    try:
        res = requests.post(RELOAD_ENDPOINT)
        if res.status_code == 200:
            print(f"[{time.ctime()}] ✅ Reload triggered successfully:", res.json())
            return True
        elif res.status_code == 409:
            print(f"[{time.ctime()}] ⚠️ Reload conflict:", res.json())
        else:
            print(f"[{time.ctime()}] ❌ Reload failed:", res.status_code, res.json())
        return False
    except Exception as e:
        print(f"[{time.ctime()}] Error during reload:", e)
        return False

def monitor_until_ready():
    print(f"[{time.ctime()}] 🔁 Monitoring model status every 10s...")
    while True:
        status = get_status()
        if status == "ready":
            print(f"[{time.ctime()}] ✅ Model is READY.")
        else:
            print(f"[{time.ctime()}] ℹ️ Model status: {status}")
        time.sleep(10)


def main():
    status = get_status()
    if status == "unloaded":
        print(f"[{time.ctime()}] ⏳ Model is unloaded. Attempting to reload...")
        if trigger_reload():
            monitor_until_ready()
    elif status == "ready":
        print(f"[{time.ctime()}] ✅ Model already ready. Monitoring...")
        monitor_until_ready()
    else:
        print(f"[{time.ctime()}] ℹ️ Current model status: {status}. Monitoring...")
        monitor_until_ready()

if __name__ == "__main__":
    main()
