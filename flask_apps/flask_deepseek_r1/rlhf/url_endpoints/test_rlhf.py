import requests
import json

# --- API Configuration ---
# All requests go to the /predict endpoint. There is no /feedback.
API_URL = "https://deepseek-test-742894389221.us-central1.run.app/predict"


def get_prediction(input_data):
    """
    Calls the API to get a crop recommendation.
    Returns the raw model output string on success, None on failure.
    """
    print("--> Sending data to /predict endpoint for prediction...")
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        result = response.json()
        print("--> Received prediction from API.")
        
        raw_rejected_string = result.get("raw_response")
        if not raw_rejected_string:
            print("[ERROR] The prediction response was missing the required 'raw_response' field.")
            return None
        return raw_rejected_string

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request for prediction failed: {e}")
        return None
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse the prediction response as JSON.")
        return None

def send_corrective_feedback(prompt_object, chosen_string, rejected_string):
    """
    Sends corrective feedback to the API.
    Crucially, it sends the payload to the SAME /predict URL.
    """
    # The payload for feedback, with the 'prompt' as a JSON object.
    feedback_payload = {
        "prompt": prompt_object,
        "chosen": chosen_string,
        "rejected": rejected_string
    }
    
    print("\n--> Sending corrective feedback to /predict endpoint...")
    print("    Payload being sent:")
    print(json.dumps(feedback_payload, indent=2))
    
    try:
        response = requests.post(API_URL, json=feedback_payload)
        response.raise_for_status()
        print("\n--> Feedback logged successfully!")
        print(f"    Server Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request for feedback failed: {e}")

def main():
    """Main function to run the command-line application."""
    print("--- Crop Recommendation and Feedback Tool ---")

    # 1. This is the original input data, which will also serve as our "prompt" object.
    input_data = {
        "Temparature": 26,
        "Humidity": 52,
        "Moisture": 38,
        "Soil Type": "Loamy",
        "Nitrogen": 37,
        "Potassium": 0,
        "Phosphorous": 0
    }

    # 2. Get the prediction
    rejected_string = get_prediction(input_data)

    if not rejected_string:
        print("\nCould not retrieve a prediction. Exiting.")
        return

    print("\n" + "="*50)
    print("         MODEL'S RECOMMENDATION")
    print("="*50)
    print(rejected_string)
    print("="*50)

    # 3. Ask the user for feedback
    while True:
        choice = input("\nIs this recommendation correct? (y/n): ").lower().strip()
        if choice in ['y', 'n']:
            break
        print("[INVALID INPUT] Please enter 'y' for yes or 'n' for no.")

    if choice == 'y':
        print("\nGreat! No feedback needed. Exiting.")
        return

    # 4. If incorrect, get the correct answer from the user
    print("\nPlease provide the correct recommendation.")
    correct_crop = input("Enter the correct Crop Type: ").strip()
    correct_fertilizer = input("Enter the correct Fertilizer Name: ").strip()

    chosen_string = (
        f"Recommended Crop Type: {correct_crop}\n"
        f"Recommended Fertilizer: {correct_fertilizer}"
    )

    # 5. Send the feedback. Note we are passing the original `input_data` dictionary.
    send_corrective_feedback(
        prompt_object=input_data,
        chosen_string=chosen_string,
        rejected_string=rejected_string
    )
    
    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()