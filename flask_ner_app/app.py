from flask import Flask, render_template, request, redirect, url_for, session
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re # re is not explicitly used in the final version here, but good to have if complex string ops were needed.

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management

# --- Model and Configuration Constants (from notebook) ---
MODEL_ID = "aryan6637/ner_training_with_values"

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
def index():
    session.clear()  # Clear any previous data
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_sentence():
    sentence = request.form.get('sentence', '').strip()
    
    if not sentence:
        # Handle empty sentence submission from the form, perhaps with a flash message
        return redirect(url_for('index')) 

    extracted_params, missing_params_info = extract_and_identify_missing(sentence)
    
    session['extracted_params'] = extracted_params
    session['missing_params_info'] = missing_params_info

    if not missing_params_info:  # All parameters were extracted by NER
        session['final_params'] = extracted_params
        return redirect(url_for('show_results'))
    else:
        # Some parameters are missing, ask the user
        return render_template('ask_missing.html', 
                               extracted_params=extracted_params, 
                               missing_params=missing_params_info)

@app.route('/submit_missing', methods=['POST'])
def submit_missing():
    # Retrieve stored data from session
    extracted_params = session.get('extracted_params', {})
    missing_params_info = session.get('missing_params_info', {}) # Details of what was missing
    
    final_parameters = extracted_params.copy() # Start with already NER-extracted params
    form_errors = {} # To collect validation errors

    # Process user input for each missing parameter
    for param_name, info in missing_params_info.items():
        user_input_str = request.form.get(param_name)
        expected_type_str = info['type'] # 'float', 'int', or 'str'
        
        if user_input_str is None or user_input_str.strip() == "":
            form_errors[param_name] = f"This field is required."
            continue

        try:
            # Convert and validate input based on expected type
            if expected_type_str == 'float':
                value = float(user_input_str)
            elif expected_type_str == 'int':
                value = int(float(user_input_str)) # Handles "30.0" then int
            else: # str
                value = user_input_str.strip()
            
            # Additional validation (e.g., non-negative for numbers as in notebook)
            if expected_type_str != 'str' and value < 0:
                form_errors[param_name] = f"{param_name.capitalize()} cannot be negative."
                continue
            
            if not value and expected_type_str == 'str': # Ensure non-empty string for soil type
                 form_errors[param_name] = f"Soil Type cannot be empty."
                 continue

            final_parameters[param_name] = value
        except ValueError:
            form_errors[param_name] = f"Invalid input. Expected a {expected_type_str}."

    if form_errors:
        # If there are errors, re-render the 'ask_missing' form with error messages
        # and previously submitted values to allow correction.
        return render_template('ask_missing.html', 
                               extracted_params=extracted_params, 
                               missing_params=missing_params_info, 
                               errors=form_errors,
                               form_values=request.form) # Pass current form values to repopulate

    # Ensure all REQUIRED_PARAMS keys are in final_parameters
    # This is a safeguard; ideally, form_errors should catch all issues.
    for req_key in REQUIRED_PARAMS.keys():
        if req_key not in final_parameters:
             # This situation implies a parameter was expected but not processed,
             # potentially due to logic error or if it wasn't in missing_params_info
             # but also not in extracted_params.
             if req_key not in form_errors: # Add error if not already present
                form_errors[req_key] = f"{req_key.capitalize()} is still missing. Please provide it."
    
    if form_errors: # Re-check if safeguard added errors
        return render_template('ask_missing.html', 
                               extracted_params=extracted_params, 
                               missing_params=missing_params_info, 
                               errors=form_errors,
                               form_values=request.form)

    session['final_params'] = final_parameters
    return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    final_params = session.get('final_params')
    if final_params is None: # Should not happen if flow is correct, but good check
        return redirect(url_for('index')) 
    return render_template('results.html', final_params=final_params)

# Initialize the model when the application starts
# For production, consider Gunicorn's --preload or similar mechanisms
# to load the model once before worker processes are forked.
with app.app_context():
    init_model()

# if __name__ == '__main__':
#     app.run(debug=True) # debug=True is for development



# ... (rest of your app.py code) ...

if __name__ == '__main__':
    # This is for local development.
    # Gunicorn will be used in production.
    # app.run(debug=True, host='0.0.0.0', port=8080)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))