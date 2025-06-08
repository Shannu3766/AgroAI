import os
import time
import torch
import logging
import gc
import multiprocessing

# Set multiprocessing start method
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from flask import Flask, request, jsonify
from unsloth import FastLanguageModel

# --- Config ---
MODEL_REPO_ID = "aryan6637/deepseek-crop-fertilizer-info-v3"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# --- Global State ---
app = Flask(__name__)
model = None
tokenizer = None
model_status = "unloaded"
model_lock = multiprocessing.Lock()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helper Functions ---
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def load_model():
    global model, tokenizer, model_status

    with model_lock:
        if model_status in ["loading", "ready"]:
            logging.info(f"Model is already in status: {model_status}")
            return

        model_status = "loading"
        logging.info(f"Loading model: {MODEL_REPO_ID}")

        try:
            if model is not None:
                del model
                del tokenizer
                clear_gpu_memory()

            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_REPO_ID,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT,
            )

            if loaded_tokenizer.pad_token is None:
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

            if torch.cuda.is_available():
                loaded_model.to("cuda")
                logging.info("Model loaded to GPU.")
            else:
                logging.info("Running on CPU.")

            model = loaded_model
            tokenizer = loaded_tokenizer
            model_status = "ready"
            logging.info("Model ready.")
        except Exception as e:
            logging.error(f"Model loading failed: {e}", exc_info=True)
            model_status = "error"
            clear_gpu_memory()

def format_prompt(params):
    return f"""### Instruction:
Given the following soil and environmental parameters:
- Temperature: {params.get('Temparature')}°C
- Humidity: {params.get('Humidity')}%
- Moisture: {params.get('Moisture')}
- Soil Type: {params.get('Soil Type')}
- Nitrogen: {params.get('Nitrogen')} ppm
- Potassium: {params.get('Potassium')} ppm
- Phosphorous: {params.get('Phosphorous')} ppm

Predict the suitable Crop Type and Fertilizer Name, and provide brief information about how they work or their characteristics.

### Response:
"""

def parse_response(text):
    result = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result

# --- Routes ---
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": model_status})

@app.route('/reload', methods=['POST'])
def reload_model():
    try:
        load_model()
        return jsonify({"message": "Model reloaded."}), 200
    except Exception as e:
        logging.error(f"Reload failed: {e}")
        return jsonify({"error": "Model reload failed"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model_status

    if model_status == "loading":
        return jsonify({"message": "Model is still loading.", "status": "LOADING"}), 503
    if model_status != "ready":
        return jsonify({"error": f"Model is not ready. Current status: {model_status}"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    required_fields = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing parameters", "required": required_fields}), 400

    try:
        with model_lock:
            clear_gpu_memory()
            prompt = format_prompt(data)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Response:" in output_text:
                output_text = output_text.split("### Response:")[1].strip()

            return jsonify({
                "input_parameters": data,
                "recommendation": parse_response(output_text),
                "raw_response": output_text
            })
    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)
        clear_gpu_memory()
        return jsonify({"error": "Inference failed"}), 500

# --- Initialize Model on Startup ---
def init_model_background():
    logging.info("Spawning model load in background process.")
    p = multiprocessing.Process(target=load_model)
    p.start()

init_model_background()

# --- Run the app ---
if __name__ == '__main__':
    while model_status == "loading":
        logging.info("Waiting for model to load...")
        time.sleep(1)
    app.run(host='0.0.0.0', port=8080, threaded=False, processes=1)
