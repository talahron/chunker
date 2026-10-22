FROM me-west1-docker.pkg.dev/moj-gcp-shared-devops/green/global-interpreter:v1@sha256:7fe95abe6dc446294b121dc9adfa37775f2cafda5595f22f167c5e1ff38940d8
WORKDIR /app
RUN uv pip install --system --break-system-packages --no-cache \
    google-cloud-pubsub==2.29.0 PyJWT==2.14.0 urllib3==2.8.0 \
    && apt-get purge -y linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -c "import pydantic_ai; from google.cloud import pubsub_v1"
COPY . /app
ENTRYPOINT ["python3", "main.py"]
