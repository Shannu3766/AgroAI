# Smart Agriculture NLP Suite

This repository contains a suite of applications and notebooks for advanced Natural Language Processing (NLP) in the agriculture domain, including Named Entity Recognition (NER), DeepSeek model integration, and web-based deployment using Flask and Docker.

---

## Project Structure

```
nlp/
├── flask_apps/
│   ├── keepactive.py
│   ├── flask_ner_app/
│   │   ├── app.py
│   │   ├── app-HP.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app.yaml
│   │   ├── .gcloudignore
│   │   ├── .dockerignore
│   │   ├── deploy.bat
│   │   ├── create_repo.bat
│   │   ├── build.bat
│   │   ├── intialize.bat
│   │   └── templates/
│   │       ├── index.html
│   │       ├── predict.html
│   │       ├── results.html
│   │       ├── enter_text.html
│   │       ├── ask_missing.html
│   │       ├── index-HP.html
│   │       ├── predict-HP.html
│   │       └── results-HP.html
│   └── flask_deepseek_r1/
│       ├── app.py
│       ├── generate_response.py
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── create_docker_gpu.bat
│       ├── rundocker.bat
│       └── rebuild.bat
├── training/
│   ├── ner-trainingwithvalues.ipynb
│   ├── ner-trainingwithvalues-HP.ipynb
│   ├── finetuning_deepseek.ipynb
│   ├── finetuning_deepseek-HP.ipynb
│   ├── ner-training-withoutvalues.ipynb
│   └── ner-training-withoutvalues-HP.ipynb
└── testing/
    ├── deepseek_with_gpu_test.ipynb
    ├── deepseek_with_gpu_test-HP.ipynb
    ├── Ner_testing_with_values.ipynb
    ├── Ner_testing_with_values-HP.ipynb
    ├── deepseek_ner_final.ipynb
    ├── deepseek_ner_final-HP.ipynb
    ├── final-testing-cpu.ipynb
    ├── final-testing-cpu-HP.ipynb
    ├── testing-bert.ipynb
    ├── testing-bert-HP.ipynb
    ├── testing-deepseek.ipynb
    ├── testing-deepseek-HP.ipynb
```

---

## Components

### 1. Flask Web Applications

#### A. `flask_ner_app`
- **Purpose:** Web interface for NER tasks.
- **Key Files:**
  - `app.py`, `app-HP.py`: Main Flask app scripts.
  - `Dockerfile`: Containerization for deployment.
  - `requirements.txt`: Python dependencies.
  - `app.yaml`, `.gcloudignore`, `.dockerignore`: Google Cloud deployment configs.
  - `deploy.bat`, `create_repo.bat`, `build.bat`, `intialize.bat`: Automation scripts.
  - `templates/`: HTML templates for the web UI.

#### B. `flask_deepseek_r1`
- **Purpose:** Web service for DeepSeek model inference.
- **Key Files:**
  - `app.py`: Main Flask app.
  - `generate_response.py`: Model response logic.
  - `Dockerfile`: Containerization for deployment.
  - `requirements.txt`: Python dependencies.
  - `create_docker_gpu.bat`, `rundocker.bat`, `rebuild.bat`: Automation scripts.

#### C. `keepactive.py`
- **Purpose:** Utility script (details depend on implementation).

---

### 2. Jupyter Notebooks

#### A. `training/`
- **Purpose:** Model training and fine-tuning.
- **Notebooks:**
  - `ner-trainingwithvalues.ipynb`, `ner-trainingwithvalues-HP.ipynb`
  - `finetuning_deepseek.ipynb`, `finetuning_deepseek-HP.ipynb`
  - `ner-training-withoutvalues.ipynb`, `ner-training-withoutvalues-HP.ipynb`

#### B. `testing/`
- **Purpose:** Model evaluation and testing.
- **Notebooks:**
  - `deepseek_with_gpu_test.ipynb`, `deepseek_with_gpu_test-HP.ipynb`
  - `Ner_testing_with_values.ipynb`, `Ner_testing_with_values-HP.ipynb`
  - `deepseek_ner_final.ipynb`, `deepseek_ner_final-HP.ipynb`
  - `final-testing-cpu.ipynb`, `final-testing-cpu-HP.ipynb`
  - `testing-bert.ipynb`, `testing-bert-HP.ipynb`
  - `testing-deepseek.ipynb`, `testing-deepseek-HP.ipynb`

---

## Deployment

### Google Cloud Run (Recommended)

1. **Build and Deploy DeepSeek Service First**
   - Go to `flask_apps/flask_deepseek_r1`
   - Edit the deployment scripts to set your Google Cloud project ID.
   - Run `create_docker_gpu.bat` or use the Dockerfile for deployment.
   - Note the service URL (e.g., `https://deepseek-flask-gpu-service-xxxxxx.run.app`).

2. **Build and Deploy NER Service**
   - Go to `flask_apps/flask_ner_app`
   - Update the NER app configuration to point to the DeepSeek service URL.
   - Run `deploy.bat` or use the Dockerfile for deployment.

### Manual Docker Deployment

- Build and run each service locally:
  ```bash
  docker build -t flask-ner-app ./flask_apps/flask_ner_app
  docker run -p 5000:5000 flask-ner-app

  docker build -t flask-deepseek-app ./flask_apps/flask_deepseek_r1
  docker run -p 5001:5000 flask-deepseek-app
  ```

---

## Usage

- Access the NER web interface via the deployed URL or `localhost:5000`.
- The NER app will communicate with the DeepSeek service for advanced inference.
- Use the Jupyter notebooks for training and testing your models.

---

## Customization

- **Project ID:** Edit the deployment scripts (`*.bat`) and `app.yaml` to set your Google Cloud project ID.
- **Service URLs:** Update the NER app configuration to point to the correct DeepSeek service endpoint after deployment.

---

## Requirements

- Python 3.8+
- pip
- Docker
- Google Cloud SDK (for cloud deployment)
- Jupyter Notebook (for running notebooks)

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/nlp](https://github.com/yourusername/nlp) 