gcloud builds submit . --tag %IMAGE_URI% --machine-type=E2_HIGHCPU_8

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