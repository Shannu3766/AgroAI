from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re # re is not explicitly used in the final version here, but good to have if complex string ops were needed.
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management

# --- Model and Configuration Constants (from notebook) ---
MODEL_ID = "aryan6637/ner_training_with_values"

# Service URLs and Status
STATUS_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/status"
RELOAD_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/reload"
PREDICT_URL = "https://deepseek-flask-gpu-service-742894389221.us-central1.run.app/predict"
SERVICE_STATUS = "unknown"  # Global variable to store service status

def update_service_status():
    global SERVICE_STATUS
    try:
        response = requests.get(STATUS_URL)
        if response.status_code == 200:
            status_data = response.json()
            SERVICE_STATUS = status_data.get('status', 'unknown')
            return True
        return False
    except Exception as e:
        print(f"Error updating service status: {str(e)}")
        return False

REQUIRED_PARAMS = {
    'Temparature': {'ner_tag': 'TEMPERATURE_VALUE', 'type': float, 'prompt': 'Enter Temperature (°C): '},
    'Humidity': {'ner_tag': 'HUMIDITY_VALUE', 'type': float, 'prompt': 'Enter Humidity (%): '},
    'Moisture': {'ner_tag': 'MOISTURE_VALUE', 'type': float, 'prompt': 'Enter Moisture (units): '},
    'Soil Type': {'ner_tag': 'SOIL_TYPE', 'type': str, 'prompt': 'Enter Soil Type: '},
    'Nitrogen': {'ner_tag': 'NITROGEN_VALUE', 'type': int, 'prompt': 'Enter Nitrogen (ppm): '},
    'Potassium': {'ner_tag': 'POTASSIUM_VALUE', 'type': int, 'prompt': 'Enter Potassium (ppm): '},
    'Phosphorous': {'ner_tag': 'PHOSPHOROUS_VALUE', 'type': int, 'prompt': 'Enter Phosphorous (ppm): '},
}

# --- Helper Functions (from notebook) ---
def clean_numeric_string(value_str):
    if value_str is None:
        return None
    # Ensure value_str is a string before replacing
    return str(value_str).replace('%', '').replace('°C', '').replace('C', '').replace('units', '').replace('ppm', '').replace(',', '').strip()

def clean_soil_type_string(value_str):
    if value_str is None:
        return None
    # Ensure value_str is a string
    return str(value_str).strip().rstrip(',.').strip()

# --- Global Model Variables ---
# These will be initialized by init_model()
tokenizer_bert_global = None
bert_model_global = None
nlp_global = None

def init_model():
    global tokenizer_bert_global, bert_model_global, nlp_global
    if nlp_global is None:  # Load only once
        print("Loading NER model (this may take a moment)...")
        try:
            tokenizer_bert_global = AutoTokenizer.from_pretrained(MODEL_ID)
            bert_model_global = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
            
            # Determine device: GPU if available, else CPU
            nlp_device = 0 if torch.cuda.is_available() else -1
            
            nlp_global = pipeline(
                "ner",
                model=bert_model_global,
                tokenizer=tokenizer_bert_global,
                aggregation_strategy="simple", # Combines B- and I- tags
                device=nlp_device
            )
            device_name = "GPU" if nlp_device == 0 else "CPU"
            print(f"NER pipeline initialized successfully on {device_name}.")
        except Exception as e:
            print(f"Error loading NER model: {e}")
            # Fallback or raise error if model loading is critical
            # For now, nlp_global will remain None, and subsequent calls will fail gracefully or be handled.
            # Consider exiting or providing a clear error message to the user if the app can't function.
    else:
        print("NER model already loaded.")

# --- Core Logic Function (adapted from notebook's get_parameters) ---
def extract_and_identify_missing(sentence):
    if nlp_global is None:
        # Handle case where model failed to load
        # All parameters will be considered missing
        missing_prompts = {
            param_name: {'prompt': details['prompt'], 'type': details['type'].__name__}
            for param_name, details in REQUIRED_PARAMS.items()
        }
        return {}, missing_prompts

    if not sentence or not sentence.strip():
        # All parameters are missing if sentence is empty
        missing_prompts = {
            param_name: {'prompt': details['prompt'], 'type': details['type'].__name__}
            for param_name, details in REQUIRED_PARAMS.items()
        }
        return {}, missing_prompts

    ner_results = nlp_global(sentence)
    extracted_raw_values = {}

    for entity in ner_results:
        # entity_group is the type of entity (e.g., 'TEMPERATURE_VALUE')
        # word is the extracted text (e.g., '30 C')
        entity_tag = entity['entity_group']
        
        param_name_from_tag = None
        for req_param_name, details in REQUIRED_PARAMS.items():
            if details['ner_tag'] == entity_tag:
                param_name_from_tag = req_param_name
                break
        
        # Store the first occurrence of a recognized entity type
        if param_name_from_tag and param_name_from_tag not in extracted_raw_values:
            extracted_raw_values[param_name_from_tag] = entity['word']

    current_parameters = {}
    missing_parameter_prompts = {}

    for param_name, details in REQUIRED_PARAMS.items():
        expected_type = details['type']
        prompt_text = details['prompt']
        raw_value_from_ner = extracted_raw_values.get(param_name)
        processed_value = None

        if raw_value_from_ner:
            cleaned_value_str = clean_numeric_string(raw_value_from_ner) if expected_type in [float, int] else clean_soil_type_string(raw_value_from_ner)
            try:
                if cleaned_value_str: # Ensure not empty after cleaning
                    if expected_type == float:
                        processed_value = float(cleaned_value_str)
                    elif expected_type == int:
                        # Convert to float first to handle "30.0", then to int
                        processed_value = int(float(cleaned_value_str)) 
                    else:  # str
                        processed_value = cleaned_value_str
            except (ValueError, TypeError):
                # Value couldn't be converted (e.g., "abc" to float)
                processed_value = None 

        # A parameter is considered successfully processed if processed_value is not None.
        # For strings, an additional check ensures it's not an empty string.
        if processed_value is not None and \
           (expected_type != str or (expected_type == str and processed_value.strip())):
            current_parameters[param_name] = processed_value
        else:
            # Parameter is missing or couldn't be processed from NER results
            missing_parameter_prompts[param_name] = {
                'prompt': prompt_text, 
                'type': expected_type.__name__ # 'float', 'int', 'str'
            }
            
    return current_parameters, missing_parameter_prompts

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET'])
def analyze():
    session.clear()  # Clear any previous data
    return render_template('enter_text.html')

@app.route('/process', methods=['POST'])
def process_sentence():
    sentence = request.form.get('sentence', '').strip()
    
    if not sentence:
        # Handle empty sentence submission from the form, perhaps with a flash message
        return redirect(url_for('analyze')) 

    extracted_params, missing_params_info = extract_and_identify_missing(sentence)
    
    # Store both extracted and missing parameters in session
    session['extracted_params'] = extracted_params
    session['missing_params_info'] = missing_params_info

    # Always render predict.html with both extracted and missing parameters
    return render_template('predict.html', 
                         extracted_params=extracted_params,
                         missing_params=missing_params_info,
                         PREDICT_URL=PREDICT_URL)

@app.route('/submit_missing', methods=['POST'])
def submit_missing():
    # Retrieve stored data from session
    extracted_params = session.get('extracted_params', {})
    missing_params_info = session.get('missing_params_info', {})
    
    final_parameters = extracted_params.copy()
    form_errors = {}

    # Process user input for each missing parameter
    for param_name, info in missing_params_info.items():
        user_input_str = request.form.get(param_name)
        expected_type_str = info['type']
        
        if user_input_str is None or user_input_str.strip() == "":
            form_errors[param_name] = f"This field is required."
            continue

        try:
            if expected_type_str == 'float':
                value = float(user_input_str)
            elif expected_type_str == 'int':
                value = int(float(user_input_str))
            else: # str
                value = user_input_str.strip()
            
            if expected_type_str != 'str' and value < 0:
                form_errors[param_name] = f"{param_name.capitalize()} cannot be negative."
                continue
            
            if not value and expected_type_str == 'str':
                 form_errors[param_name] = f"Soil Type cannot be empty."
                 continue

            final_parameters[param_name] = value
        except ValueError:
            form_errors[param_name] = f"Invalid input. Expected a {expected_type_str}."

    if form_errors:
        return render_template('predict.html', 
                             extracted_params=extracted_params,
                             missing_params=missing_params_info,
                             errors=form_errors,
                             form_values=request.form,
                             PREDICT_URL=PREDICT_URL)

    # Update session with final parameters
    session['final_params'] = final_parameters
    
    # Return to predict page with updated parameters
    return render_template('predict.html', 
                         extracted_params=final_parameters,
                         missing_params={},  # No missing params after successful submission
                         PREDICT_URL=PREDICT_URL)

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    try:
        data = request.get_json()
        if not data:
            print("Error: No data provided in request")
            return jsonify({'error': 'No data provided'}), 400

        print("\nSending data to prediction service:")
        print("Request data:", json.dumps(data, indent=2))

        # Make request to prediction service
        response = requests.post(PREDICT_URL, json=data)
        
        print("\nResponse from prediction service:")
        print("Status code:", response.status_code)
        
        if response.status_code == 200:
            prediction_result = response.json()
            print("Prediction result:", json.dumps(prediction_result, indent=2))
            
            # Store the result in session
            session['prediction_result'] = prediction_result
            
            # Return the formatted response
            return jsonify({
                'recommendation': prediction_result.get('recommendation', {}),
                'input_parameters': prediction_result.get('input_parameters', {}),
                'raw_response': prediction_result.get('raw_response', '')
            })
        else:
            print("Error response:", response.text)
            return jsonify({'error': 'Prediction service error'}), 500

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def show_results():
    final_params = session.get('final_params')
    prediction_result = session.get('prediction_result')
    
    if final_params is None: # Should not happen if flow is correct, but good check
        return redirect(url_for('index'))
        
    return render_template('results.html', 
                         final_params=final_params,
                         prediction_result=prediction_result)

@app.route('/check_status', methods=['GET'])
def check_status():
    global SERVICE_STATUS
    try:
        if update_service_status():
            if SERVICE_STATUS == 'unloaded':
                # Trigger reload
                reload_response = requests.post(RELOAD_URL)
                if reload_response.status_code == 200:
                    SERVICE_STATUS = 'ready'  # Update status after successful reload
                    return {'message': 'Service reloaded successfully', 'status': SERVICE_STATUS}, 200
                else:
                    return {'error': 'Failed to reload service', 'status': SERVICE_STATUS}, 500
            elif SERVICE_STATUS == 'ready':
                return {'message': 'Service is ready', 'status': SERVICE_STATUS}, 200
            else:
                return {'error': 'Unknown status', 'status': SERVICE_STATUS}, 500
        else:
            return {'error': 'Failed to check status', 'status': SERVICE_STATUS}, 500
    except Exception as e:
        return {'error': f'Error checking status: {str(e)}', 'status': SERVICE_STATUS}, 500


with app.app_context():
    # Check service status before initializing
    if update_service_status():
        if SERVICE_STATUS == 'unloaded':
            # Trigger reload
            reload_response = requests.post(RELOAD_URL)
            if reload_response.status_code == 200:
                SERVICE_STATUS = 'ready'
                print("Service reloaded successfully")
            else:
                print("Warning: Failed to reload service")
        elif SERVICE_STATUS == 'ready':
            print("Service is ready")
        else:
            print(f"Warning: Unknown service status: {SERVICE_STATUS}")
    else:
        print("Warning: Failed to check service status")
    
    # Initialize the model
    init_model()


if __name__ == '__main__':
    # This is for local development.
    # Gunicorn will be used in production.
    # app.run(debug=True, host='0.0.0.0', port=8080)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))