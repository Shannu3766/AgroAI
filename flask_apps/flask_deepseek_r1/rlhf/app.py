import os
import time
import torch
import logging
import gc
import multiprocessing
import pandas as pd
from datetime import datetime

# --- Load .env for local development ---
from dotenv import load_dotenv
load_dotenv()

# --- PyMongo for Database ---
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- Flask for API ---
from flask import Flask, request, jsonify

# --- Unsloth and TRL for Model & Training ---
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# =====================================================================================
# CONFIGURATION (from Environment Variables)
# =====================================================================================
INITIAL_MODEL_REPO_ID = os.environ.get("INITIAL_MODEL_REPO_ID", "aryan6637/deepseek-crop-fertilizer-dpo")
NEW_MODEL_REPO_TEMPLATE = os.environ.get("NEW_MODEL_REPO_TEMPLATE", "aryan6637/deepseek-crop-fertilizer-dpo-v{version}")
HF_TOKEN = os.environ.get("HF_TOKEN")
MONGO_URI = os.environ.get("MONGO_URI")
RETRAINING_THRESHOLD = int(os.environ.get("RETRAINING_THRESHOLD", 50))

MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
DB_NAME = "llm_feedback_db"
COLLECTION_NAME = "dpo_feedback"

# --- Centralized list of keys required by format_prompt ---
INPUT_PARAMETER_KEYS = [
    'Temparature', 'Humidity', 'Moisture', 'Soil Type',
    'Nitrogen', 'Potassium', 'Phosphorous'
]

# =====================================================================================
# SETUP & GLOBAL STATE
# =====================================================================================
try:
    multiprocessing.set_start_method('spawn', force=True)
    logging.info("Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    pass

app = Flask(__name__)
manager = multiprocessing.Manager()
shared_state = manager.dict()
shared_state['model_status'] = "unloaded"
shared_state['current_model_id'] = INITIAL_MODEL_REPO_ID
model_lock = manager.Lock()

model, tokenizer, db_client = None, None, None
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if not HF_TOKEN:
    logging.warning("HF_TOKEN environment variable not set. Pushing new models to the Hub will fail.")
if not MONGO_URI:
    logging.error("MONGO_URI environment variable not set. Database functionality will fail.")


# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================
def get_db_client():
    global db_client
    if db_client: return db_client
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logging.info("Successfully connected to MongoDB.")
        db_client = client
        return db_client
    except (ConnectionFailure, AttributeError) as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None

def clear_gpu_memory():
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    gc.collect()

def format_prompt(params):
    """
    MODIFIED: Generates the full prompt text from a dictionary of parameters
    using a clear instruction-following format. This guides the model to provide a
    direct answer instead of generating code.
    NOTE: The keys used here must match the `INPUT_PARAMETER_KEYS` list.
    """
    return f"""### Instruction:
Given the following soil and environmental parameters, predict the suitable Crop Type and Fertilizer Name.

- Temperature: {params.get('Temparature')}°C
- Humidity: {params.get('Humidity')}%
- Moisture: {params.get('Moisture')}%
- Soil Type: {params.get('Soil Type')}
- Nitrogen: {params.get('Nitrogen')} ppm
- Potassium: {params.get('Potassium')} ppm
- Phosphorous: {params.get('Phosphorous')} ppm

### Response:
"""


def parse_response(text):
    """
    MODIFIED: Parses the model's raw output to extract key-value pairs.
    It is designed to handle the clean output from the model.
    """
    result = {}
    # The /predict endpoint now provides only the response, but this check
    # makes the function robust if it's ever used with the full text.
    if "### Response:" in text:
        text = text.split("### Response:")[1].strip()

    for line in text.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result

def get_next_model_version(current_id):
    try:
        base_name, version_str = current_id.rsplit('-v', 1)
        next_version = int(version_str) + 1
        return NEW_MODEL_REPO_TEMPLATE.format(version=next_version)
    except (ValueError, IndexError):
        # Handles case where the initial model doesn't have a version suffix
        return NEW_MODEL_REPO_TEMPLATE.format(version=2)


def load_model():
    global model, tokenizer
    with model_lock:
        if shared_state['model_status'] != "unloaded":
            logging.info(f"Model load skipped. Status: {shared_state['model_status']}")
            return
        shared_state['model_status'] = "loading"
        model_id_to_load = shared_state['current_model_id']
        logging.info(f"Loading model: {model_id_to_load}")
        try:
            if model is not None: del model; del tokenizer; clear_gpu_memory()
            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id_to_load, max_seq_length=MAX_SEQ_LENGTH,
                dtype=None, load_in_4bit=LOAD_IN_4BIT,
            )
            if loaded_tokenizer.pad_token is None:
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
            model, tokenizer = loaded_model, loaded_tokenizer
            shared_state['model_status'] = "ready"
            logging.info(f"Model '{model_id_to_load}' is ready.")
        except Exception as e:
            logging.error(f"Model loading failed: {e}", exc_info=True)
            shared_state['model_status'] = "error"
            clear_gpu_memory()

def run_retraining_job():
    with model_lock:
        shared_state['model_status'] = "retraining"
        logging.info("RETRAINING JOB STARTED. API will not serve predictions.")
        clear_gpu_memory()
    client = get_db_client()
    if not client:
        logging.error("Retraining failed: Cannot connect to DB."); shared_state['model_status'] = "error"
        return
    try:
        db = client[DB_NAME]
        feedback_collection = db[COLLECTION_NAME]
        archive_collection = db[f"{COLLECTION_NAME}_archive"]
        feedback_data = list(feedback_collection.find({}))
        if not feedback_data:
            logging.warning("Retraining aborted: no new feedback data found."); shared_state['model_status'] = "ready"
            # Since retraining is aborted, we must load the model again if it was unloaded
            load_model()
            return
        logging.info(f"Fetched {len(feedback_data)} entries for retraining.")
        new_preference_dataset = Dataset.from_pandas(pd.DataFrame(feedback_data))
        
        global model, tokenizer
        if model is None or tokenizer is None:
             logging.info("Model not loaded in retraining process, loading it now."); load_model()
        
        model.train()
        dpo_config = DPOConfig(
            beta=0.1, output_dir="./dpo_retraining_output", per_device_train_batch_size=1,
            gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=1, learning_rate=5e-5,
            fp16=True, logging_steps=5, optim="adamw_8bit", report_to="none", remove_unused_columns=False,
        )
        dpo_trainer = DPOTrainer(
            model=model, args=dpo_config, train_dataset=new_preference_dataset, tokenizer=tokenizer,
            format_dataset=lambda ex: {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
        )
        logging.info("Starting DPO training..."); dpo_trainer.train(); logging.info("DPO training finished.")
        new_model_id = get_next_model_version(shared_state['current_model_id'])
        logging.info(f"Pushing newly trained model to Hub: {new_model_id}")
        dpo_trainer.model.push_to_hub(new_model_id, token=HF_TOKEN); tokenizer.push_to_hub(new_model_id, token=HF_TOKEN)
        logging.info("Successfully pushed to Hugging Face Hub.")
        
        if feedback_data:
            archive_collection.insert_many(feedback_data); feedback_collection.delete_many({})
            logging.info(f"Archived and removed {len(feedback_data)} feedback entries.")
        
        with model_lock:
            shared_state['current_model_id'] = new_model_id
            shared_state['model_status'] = "unloaded"
        logging.info(f"Retraining complete. System will now load new model: {new_model_id}")
    except Exception as e:
        logging.error(f"Retraining job failed: {e}", exc_info=True)
        with model_lock: shared_state['model_status'] = "error"
    finally:
        clear_gpu_memory();
        if model: model.eval()

# =====================================================================================
# FLASK API ROUTES
# =====================================================================================
@app.route('/status', methods=['GET'])
def status():
    """Returns the current status of the model."""
    if shared_state['model_status'] == "unloaded":
         logging.info("Status check triggered post-training model reload."); load_model()
    return jsonify({
        "status": shared_state['model_status'],
        "current_model_id": shared_state['current_model_id'],
        "retraining_threshold": RETRAINING_THRESHOLD,
    })

@app.route('/reload', methods=['POST'])
def reload_model_endpoint():
    """Manually triggers a model reload."""
    current_status = shared_state['model_status']
    if current_status in ["loading", "retraining"]:
        return jsonify({"message": f"Cannot reload, status is '{current_status}'."}), 409
    logging.info("Manual model reload triggered via /reload endpoint.")
    with model_lock: shared_state['model_status'] = "unloaded"
    load_model()
    if shared_state['model_status'] == 'ready':
        return jsonify({"message": "Model reloaded successfully.", "status": shared_state['model_status']}), 200
    else:
        return jsonify({"error": "Model reload failed. Check logs.", "status": shared_state['model_status']}), 500

@app.route('/retrain', methods=['POST'])
def manual_retrain_endpoint():
    """
    Manually triggers a retraining job using all available feedback data,
    regardless of the retraining threshold.
    """
    current_status = shared_state['model_status']
    if current_status in ["loading", "retraining"]:
        return jsonify({ "message": f"Cannot start retraining, system is busy. Current status: '{current_status}'." }), 409

    client = get_db_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500

    try:
        feedback_count = client[DB_NAME][COLLECTION_NAME].count_documents({})
        if feedback_count == 0:
            return jsonify({"message": "Retraining not started: no new feedback data available."}), 400

        logging.info(f"MANUAL retraining triggered via /retrain endpoint with {feedback_count} entries.")
        p = multiprocessing.Process(target=run_retraining_job)
        p.start()

        return jsonify({
            "message": f"Retraining job started successfully with {feedback_count} feedback entries.",
            "status_after_request": "retraining"
        }), 202
    except Exception as e:
        logging.error(f"Failed to start manual retraining: {e}", exc_info=True)
        return jsonify({"error": "Failed to start manual retraining job."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Generates a prediction based on input parameters."""
    status = shared_state['model_status']
    if status in ["loading", "retraining"]: return jsonify({"message": f"Model busy. Status: {status}"}), 503
    if status != "ready":
        load_model()
        if shared_state['model_status'] != 'ready': return jsonify({"error": f"Model recovery failed. Status: {shared_state['model_status']}"}), 503

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()

    if not all(field in data for field in INPUT_PARAMETER_KEYS):
        return jsonify({"error": "Missing parameters", "required": INPUT_PARAMETER_KEYS}), 400

    try:
        with model_lock:
            # The prompt is now created using the improved instruction format
            prompt_text = format_prompt(data)
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            input_ids_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=150, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.6, top_p=0.9
                )
            
            # --- MODIFIED: Decode only the newly generated tokens ---
            newly_generated_tokens = outputs[0][input_ids_len:]
            output_text = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)

            return jsonify({
                "input_parameters": data,
                "recommendation": parse_response(output_text),
                "raw_response": output_text.strip(),
                "model_version": shared_state['current_model_id'],
                "prompt_used": prompt_text
            })
    except Exception as e:
        logging.error(f"Inference failed: {e}", exc_info=True)
        return jsonify({"error": "Inference failed"}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """
    Handles user feedback for DPO. The request must contain the full 'prompt'
    text (as returned by /predict), a 'chosen' response (the preferred outcome),
    and a 'rejected' response (the model's original, incorrect output).
    """
    if shared_state['model_status'] == 'retraining':
        return jsonify({"message": "System is retraining, feedback will be logged."}), 202

    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    required_fields = ['prompt', 'chosen', 'rejected']
    if not all(field in data for field in required_fields): return jsonify({"error": "Missing parameters", "required": required_fields}), 400

    client = get_db_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500

    try:
        db = client[DB_NAME]
        db[COLLECTION_NAME].insert_one({
            "prompt": data["prompt"],
            "chosen": data["chosen"],
            "rejected": data["rejected"],
            "timestamp": datetime.utcnow(),
            "model_version": data.get("model_version", shared_state.get('current_model_id'))
        })
        current_feedback_count = db[COLLECTION_NAME].count_documents({})
        message = f"Feedback logged. Total entries: {current_feedback_count}."

        if current_feedback_count >= RETRAINING_THRESHOLD and shared_state['model_status'] != 'retraining':
            message += " Threshold reached. Triggering automatic retraining job."
            logging.info(f"AUTOMATIC retraining triggered by threshold ({current_feedback_count}/{RETRAINING_THRESHOLD}).")
            p = multiprocessing.Process(target=run_retraining_job)
            p.start()

        return jsonify({"message": message}), 201
    except Exception as e:
        logging.error(f"Failed to process feedback: {e}", exc_info=True)
        return jsonify({"error": "Failed to process feedback"}), 500

# =====================================================================================
# APPLICATION STARTUP
# =====================================================================================
def init_app():
    logging.info("Starting application initialization.")
    get_db_client()
    logging.info("Spawning initial model load.")
    # Run load_model in a separate process to avoid blocking the main thread
    # on startup, especially in a production environment with a gunicorn-like server.
    p = multiprocessing.Process(target=load_model)
    p.start()


if __name__ == '__main__':
    # When running directly, we initialize and then start the Flask app.
    # The initial load will happen in a background process.
    init_app()
    app.run(host='0.0.0.0', port=8080, threaded=False)