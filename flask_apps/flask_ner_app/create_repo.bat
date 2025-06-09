gcloud artifacts repositories create %AR_REPOSITORY_NAME% ^
  --repository-format=docker ^
  --location=%REGION% ^
  --description="Docker repo for my NER Flask app"