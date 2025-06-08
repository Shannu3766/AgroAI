@echo off
REM Set environment variables
set PROJECT_ID=axiomatic-grove-403514
set REGION=us-central1
set AR_REPOSITORY_NAME=deepseek-app-repo
set IMAGE_NAME=deepseek-flask-app
set IMAGE_TAG=latest
set SERVICE_NAME=deepseek-flask-service
set IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%AR_REPOSITORY_NAME%/%IMAGE_NAME%:%IMAGE_TAG%

REM Create Artifact Registry repository
gcloud artifacts repositories create %AR_REPOSITORY_NAME% ^
  --repository-format=docker ^
  --location=%REGION% ^
  --description="Docker repo for my deepseek Flask app"

REM Submit build to Cloud Build
@REM gcloud builds submit . --tag %IMAGE_URI%
gcloud builds submit . --tag %IMAGE_URI% --machine-type=e2-highmem-8 

REM Deploy to Cloud Run
gcloud run deploy %SERVICE_NAME% ^
    --image="%IMAGE_URI%" ^
    --platform=managed ^
    --region=%REGION% ^
    --allow-unauthenticated ^
    --port=5000 ^
    --memory=32Gi ^
    --cpu=8 ^
    --timeout=3000 ^
    --concurrency=10 ^
    --cpu-boost ^
    --min-instances=0