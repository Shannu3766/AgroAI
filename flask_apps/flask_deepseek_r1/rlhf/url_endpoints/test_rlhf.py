import requests
import json

# --- API Configuration ---
# The URL is the same for both prediction and feedback.
API_URL = "https://deepseek-test-742894389221.us-central1.run.app/predict"

def format_dict_to_string(data_dict):
    """Converts a dictionary into a formatted multi-line string."""
    return "\n".join([f"{key}: {value}" for key, value in data_dict.items()])

def get_prediction(input_data):
    """Calls the API to get a crop recommendation."""
    print("--> Sending data for initial prediction...")
    try:
        response = requests.post(API_URL, json=input_data)
        response.raise_for_status()
        full_result = response.json()
        print("--> Received prediction from API.")
        return full_result.get("recommendation")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request for prediction failed: {e}")
        return None
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse the prediction response as JSON.")
        return None

def send_corrective_feedback(prompt_string, chosen_string, rejected_string):
    """
    Sends corrective feedback to the API.
    Note: It sends the data to the SAME URL as the prediction.
    """
    feedback_payload = {
        "prompt": prompt_string,
        "chosen": chosen_string,
        "rejected": rejected_string
    }
    
    print("\n--> Sending corrective feedback to the API...")
    print("    Payload being sent:")
    print(json.dumps(feedback_payload, indent=2))
    
    try:
        response = requests.post(API_URL, json=feedback_payload)
        response.raise_for_status()
        print("\n--> Feedback logged successfully!")
        # Use .text for non-JSON responses, which is common for feedback endpoints
        print(f"    Response from server: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request for feedback failed: {e}")

def main():
    """Main function to run the command-line application."""
    print("--- Crop Recommendation and Feedback Tool ---")

    # 1. Define the initial input data
    input_data = {
        "Temparature": 26,
        "Humidity": 52,
        "Moisture": 38,
        "Soil Type": "Loamy",
        "Nitrogen": 37,
        "Potassium": 0,
        "Phosphorous": 0
    }

    # 2. Get the initial prediction from the model
    model_recommendation = get_prediction(input_data)
    if not model_recommendation:
        print("\nCould not retrieve a prediction. Exiting.")
        return

    # 3. Format ALL pieces of data (prompt, chosen, rejected) into STRINGS
    
    # Format the original input into the "prompt" string
    prompt_string = format_dict_to_string(input_data)

    # Format the model's output into the "rejected" string
    rejected_crop = model_recommendation.get("Recommended Crop Type", "N/A")
    rejected_fertilizer = model_recommendation.get("Recommended Fertilizer", "N/A")
    rejected_string = (
        f"Recommended Crop Type: {rejected_crop}\n"
        f"Recommended Fertilizer: {rejected_fertilizer}"
    )

    print("\n" + "="*50)
    print("         MODEL'S RECOMMENDATION")
    print("="*50)
    print(rejected_string)
    print("="*50)

    # 4. Ask the user if this prediction is correct
    while True:
        choice = input("\nIs this recommendation correct? (y/n): ").lower().strip()
        if choice in ['y', 'n']:
            break
        print("[INVALID INPUT] Please enter 'y' for yes or 'n' for no.")

    if choice == 'y':
        print("\nGreat! No feedback needed. Exiting.")
        return

    # 5. If incorrect, get the correct answer from the user
    print("\nPlease provide the correct recommendation.")
    correct_crop = input("Enter the correct Crop Type: ").strip()
    correct_fertilizer = input("Enter the correct Fertilizer Name: ").strip()

    # Format the user's correct answer into the "chosen" string
    chosen_string = (
        f"Recommended Crop Type: {correct_crop}\n"
        f"Recommended Fertilizer: {correct_fertilizer}"
    )

    # 6. Send the corrective feedback with all parts as strings
    send_corrective_feedback(
        prompt_string=prompt_string,
        chosen_string=chosen_string,
        rejected_string=rejected_string
    )
    
    print("\n--- Process Complete ---")

if __name__ == "__main__":
    main()