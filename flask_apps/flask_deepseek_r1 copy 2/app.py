# app.py

import os
import torch
import threading
import logging
import multiprocessing

# Set multiprocessing start method to 'spawn'
if torch.cuda.is_available():
    # Force process start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)

from flask import Flask, request, jsonify
from unsloth import FastLanguageModel

# --- Configuration ---
# Set the Hugging Face model repository ID
MODEL_REPO_ID = "aryan6637/deepseek-crop-fertilizer-info-v3"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True # Use 4-bit quantization for memory efficiency

# --- Global State ---
app = Flask(__name__)
model = None
tokenizer = None
model_status = "unloaded" # Can be "unloaded", "loading", "ready", "error"
model_lock = threading.Lock() # To prevent race conditions during reload/predict

# --- Logging Setup ---
# Configure logging to provide clear output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Core Model and Inference Logic ---

def load_model():
    """
    Loads the model and tokenizer from Hugging Face.
    This function is thread-safe.
    """
    global model, tokenizer, model_status
    
    with model_lock:
        if model_status == "loading":
            logging.warning("Model is already being loaded.")
            return

        model_status = "loading"
        logging.info(f"Attempting to load model: {MODEL_REPO_ID}...")

        try:
            # Free up VRAM if a model is already loaded
            if model is not None:
                del model
                del tokenizer
                torch.cuda.empty_cache()
                logging.info("Unloaded previous model and cleared CUDA cache.")

            # Load the model using Unsloth's FastLanguageModel
            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_REPO_ID,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,      # Autodetect
                load_in_4bit=LOAD_IN_4BIT,
            )
            
            # Ensure tokenizer has a pad token
            if loaded_tokenizer.pad_token is None:
                logging.warning("Tokenizer has no pad token; setting it to eos_token.")
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

            # Move model to GPU if available
            if torch.cuda.is_available():
                loaded_model.to("cuda")
                logging.info("Model successfully moved to GPU.")
            else:
                logging.warning("CUDA not available. Model is running on CPU.")

            # Update global state
            model = loaded_model
            tokenizer = loaded_tokenizer
            model_status = "ready"
            logging.info("Model loaded successfully and is ready for inference.")
            
        except Exception as e:
            model_status = "error"
            model = None
            tokenizer = None
            logging.error(f"Failed to load model: {e}", exc_info=True)


def format_inference_prompt(params: dict) -> str:
    """
    Formats the input parameters into the Alpaca prompt format for the model.
    """
    instruction_text = (
        f"Given the following soil and environmental parameters:\n"
        f"- Temperature: {params.get('Temparature')}°C\n"
        f"- Humidity: {params.get('Humidity')}% \n"
        f"- Moisture: {params.get('Moisture')}\n"
        f"- Soil Type: {params.get('Soil Type')}\n"
        f"- Nitrogen: {params.get('Nitrogen')} ppm\n"
        f"- Potassium: {params.get('Potassium')} ppm\n"
        f"- Phosphorous: {params.get('Phosphorous')} ppm\n\n"
        f"Predict the suitable Crop Type and Fertilizer Name, and provide brief information about how they work or their characteristics."
    )
    # The model was trained with this specific format.
    return f"### Instruction:\n{instruction_text}\n\n### Response:\n"

def parse_response(text: str) -> dict:
    """
    Parses the model's raw text output into a structured dictionary.
    """
    response_dict = {}
    lines = text.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            response_dict[key.strip()] = value.strip()
    return response_dict

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check the current status of the model."""
    return jsonify({"status": model_status})

@app.route('/reload', methods=['POST'])
def reload_model_endpoint():
    """Endpoint to trigger a redownload and reload of the model."""
    logging.info("Received request to reload the model.")
    # Run loading in a separate thread to avoid blocking the request
    reload_thread = threading.Thread(target=load_model)
    reload_thread.start()
    return jsonify({"message": "Model reload initiated. Check /status for progress."}), 202

@app.route('/predict', methods=['POST'])
def predict():
    if model_status != "ready":
        return jsonify({"error": f"Model not ready. Current status: {model_status}"}), 503

    # Ensure request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    # Validate input
    required_params = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    if not all(key in data for key in required_params):
        return jsonify({"error": "Missing one or more required parameters", "required": required_params}), 400

    with model_lock:
        try:
            # Clear CUDA cache before prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Format prompt and tokenize
            prompt = format_inference_prompt(data)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate response from model with explicit memory management
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                
                # Immediately move outputs to CPU to free GPU memory
                outputs = outputs.cpu()
                
            # Clear inputs from GPU memory
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Decode and extract the response part
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = full_response.split("### Response:")[1].strip()

            # Clear more memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Parse the structured data and return
            parsed_data = parse_response(response_only)

            return jsonify({
                "input_parameters": data,
                "recommendation": parsed_data,
                "raw_response": response_only
            })

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)
            # Ensure memory is cleared even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return jsonify({"error": "Internal server error during prediction."}), 500

# --- Main Execution ---

# ** CORRECTED SECTION **
# This code runs when the module is imported by Gunicorn.
# We start the model loading in a background thread so the server can start
# immediately and respond to /status requests while the model loads.
if model_status == "unloaded":
    logging.info("Application starting up. Initiating model load in background.")
    initial_load_thread = threading.Thread(target=load_model)
    initial_load_thread.start()

# This block is for direct execution (e.g., `python app.py`) for local testing.
if __name__ == '__main__':
    # When running directly, we wait for the model to finish loading before starting the server.
    if initial_load_thread.is_alive():
        initial_load_thread.join()
    app.run(host='0.0.0.0', port=8080)