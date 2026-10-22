import json
from pathlib import Path
from typing import Any
from functools import wraps
from google.cloud import storage
from src.logger import setup_logger
from src.exceptions import MaxRetriesAttemptsError
import asyncio  # Added for async operations
from src.pydantic_models import FileConfig

# Retry wrapper
def retry(max_retries: int = 3, delay: float = 0.2):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        await asyncio.sleep(delay)
                    run_func = await asyncio.to_thread(func, self, *args, **kwargs)
                    self.logger.info(f"Attempt {attempt} for function - {func.__name__} success")
                    return run_func
                except Exception as exc:
                    self.logger.warning(f"Attempt {attempt} failed in {func.__name__}: {exc}")
                    if attempt == max_retries:
                        raise MaxRetriesAttemptsError(f"Max retries reached")
        return wrapper
    return decorator


class GCSManager:
    """Create a GCSManager."""
    def __init__(self):
        self.client = storage.Client()
        self.logger = setup_logger(self.__class__.__name__)

    @retry(max_retries=3, delay=0.2)
    def download_json(self, bucket_name: str, base_name: str) -> tuple[Any,str]:
        """Download a blob's contents as text.
        Args:
            bucket_name: Name of the GCS bucket.
            base_name: name of the file as blob.
        Returns:
            The blob data decoded to a string, and the file name prefix
        """
        bucket = self.client.bucket(bucket_name)
        # create the file name without its suffix to use it later in the pipeline
        object_path = Path(base_name).stem
        blob = bucket.blob(base_name)
        json_text = blob.download_as_text()
        return json.loads(json_text), object_path

    async def load_file_config(
            self,
            file_name: str,
            project_name: str = "",
            index_mapping_bucket: str = "",
            index_mapping_object_prefix: str = "indexes",
            file_metadata_bucket_name: str = "",
            use_dummy_file_config: bool = False,
    ) -> FileConfig:
        """Load the JSON that holds the route of the processed file."""
        if use_dummy_file_config:
            dummy_path = Path(__file__).resolve().parent.parent / "dummy_file_config.json"
            self.logger.info(f"Using dummy file config {dummy_path.name} for {file_name}")
            file_config = FileConfig(**json.loads(dummy_path.read_text(encoding="utf-8")))
            self.logger.info(f"File config for {file_name}: {file_config}")
            return file_config

        # 1. Path in moj-gcp-d-rag-index-mapping-bucket -> indexes/{project_name}.json
        project_blob_name = f"{index_mapping_object_prefix}/{project_name}.json"

        # 2. Path in moj-gcp-d-rag-input-{project_name} -> {file_name}.json (ensuring .json suffix)
        file_blob_name = str(Path(file_name).with_suffix(".json"))

        self.logger.info(
            f"Loading file config from GCS: "
            f"project_config=gs://{index_mapping_bucket}/{project_blob_name}, "
            f"file_metadata=gs://{file_metadata_bucket_name}/{file_blob_name}"
        )

        # Download both JSONs concurrently using the existing retry-wrapped download_json
        (project_data, _), (file_data, _) = await asyncio.gather(
            self.download_json(index_mapping_bucket, project_blob_name),
            self.download_json(file_metadata_bucket_name, file_blob_name),
        )

        merged_config = {**project_data, **file_data}
        file_config = FileConfig(**merged_config)
        self.logger.info(f"File config for {file_name}: {file_config}")
        return file_config
