set PROJECT_ID=axiomatic-grove-403514
set REGION=us-central1
set AR_REPOSITORY_NAME=ner-app-repo
set IMAGE_NAME=ner-flask-app
set IMAGE_TAG=latest
set SERVICE_NAME=ner-flask-service
set IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%AR_REPOSITORY_NAME%/%IMAGE_NAME%:%IMAGE_TAG%

gcloud artifacts repositories create %AR_REPOSITORY_NAME% ^
  --repository-format=docker ^
  --location=%REGION% ^
  --description="Docker repo for my NER Flask app"

gcloud builds submit . --tag %IMAGE_URI%

gcloud run deploy %SERVICE_NAME% ^
    --image="%IMAGE_URI%" ^
    --platform=managed ^
    --region=%REGION% ^
    --allow-unauthenticated ^
    --port=8080 ^
    --memory=4Gi ^
    --cpu=4 ^
    --timeout=300 ^
    --concurrency=10 ^
    --cpu-boost ^
    --min-instances=0