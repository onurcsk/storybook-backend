run: 
	uvicorn backend:app --reload
docker_build:
	docker build -t  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/storybook/${GAR_IMAGE}:prod .
docker_run:
	docker run -e PORT=8000 -p 8000:8000 --env-file .env ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/storybook/${GAR_IMAGE}:prod