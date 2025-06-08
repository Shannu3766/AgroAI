@echo off


set PROJECT_ID=axiomatic-grove-403514
set REGION=us-central1
set AR_REPOSITORY_NAME=deepseek-app-repo
set IMAGE_NAME=deepseek-flask-app
set IMAGE_TAG=latest
set SERVICE_NAME=deepseek-flask-gpu-service1

set GPU_TYPE=nvidia-l4
set GPU_COUNT=1
set MEMORY=32Gi
set CPU=8
set PORT=8080
set TIMEOUT=900
set CONCURRENCY=1

set IMAGE_URI=%REGION%-docker.pkg.dev/%PROJECT_ID%/%AR_REPOSITORY_NAME%/%IMAGE_NAME%:%IMAGE_TAG%

gcloud config set project %PROJECT_ID%

gcloud artifacts repositories create %AR_REPOSITORY_NAME% ^
  --repository-format=docker ^
  --location=%REGION% ^
  --description="Docker repo for DeepSeek Flask app" 

gcloud builds submit . --tag %IMAGE_URI% --machine-type=E2_HIGHCPU_8


gcloud run deploy %SERVICE_NAME% ^
    --image="%IMAGE_URI%" ^
    --platform=managed ^
    --region=%REGION% ^
    --allow-unauthenticated ^
    --port=%PORT% ^
    --memory=%MEMORY% ^
    --cpu=%CPU% ^
    --timeout=%TIMEOUT% ^
    --concurrency=%CONCURRENCY% ^
    --execution-environment=gen2 ^
    --gpu=type=%GPU_TYPE%,count=%GPU_COUNT% ^
    --min-instances=0

gcloud run deploy %SERVICE_NAME% ^
  --image %IMAGE_URI% ^
  --region %REGION% ^
  --project %PROJECT_ID% ^
  --gpu=%GPU_COUNT% ^
  --gpu-type=%GPU_TYPE% ^
  --cpu=8 ^
  --memory=32Gi ^
  --port=8080 ^
  --max-instances=1 ^
  --timeout=900 ^
  --concurrency=1 ^
  --execution-environment=gen2 ^
  --allow-unauthenticated