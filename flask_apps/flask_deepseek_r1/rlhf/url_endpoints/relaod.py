import requests
import time

STATUS_ENDPOINT = "https://deepseek-test-742894389221.us-central1.run.app/status"
RELOAD_ENDPOINT = "https://deepseek-test-742894389221.us-central1.run.app/reload"

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
    
if __name__ == "__main__":
    trigger_reload()