import json
from pathlib import Path
from typing import Any
from functools import wraps
from google.cloud import storage
from src.logger import setup_logger
from src.exceptions import MaxRetriesAttemptsError
import asyncio  # Added for async operations

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
