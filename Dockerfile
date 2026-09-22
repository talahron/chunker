FROM me-west1-docker.pkg.dev/moj-gcp-shared-devops/green/global-interpreter:v1
WORKDIR /app
COPY . /app
ENTRYPOINT ["python3", "main.py"]
