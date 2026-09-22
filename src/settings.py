import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.pydantic_models import ChunkingStrategy
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GCP_PROJECT_ID: str
    GCP_LOCATION: str
    CACHING_LOCATION: str
    EMBEDDINGS_REGION: str
    INPUT_BUCKET_NAME_INIT: str
    INPUT_BUCKET_NAME_INIT_PDF: str
    INPUT_BUCKET_NAME: str = None
    INPUT_BUCKET_NAME_PDF: str = None
    PROJECT_NAME: str
    FILE_NAME: str
    FILE_NAME_PDF: str = None
    MODEL_NAME: str
    MODEL_NAME_ENRICHMENT_CHUNKER: str
    THINKING_BUDGET_ENRICHMENT_CHUNKER: int
    MAX_TOKENS: int
    MIN_TOKENS: int
    INDEX_MAPPING_BUCKET:str
    VECTOR_BUCKET_NAME: str
    INDEX_MAPPING_JSON_FILE: str
    EMBEDDINGS_MODEL_NAME: str
    INDEX_REGION: str
    VECTOR_BUCKET_NAME_INIT: str
    VECTOR_BUCKET_NAME: str = None
    KNOWLEDGEBASE_BUCKET_NAME_INIT: str
    KNOWLEDGEBASE_BUCKET_NAME: str = ""
    CHUNKING_STRATEGY: ChunkingStrategy
    INCREMENT: int
    BATCH_SIZE: int
    MODEL_NAME_SENTENCE_SPLITTER: str
    MAX_WORKERS_FOR_BATCH_PROCESSING: int
    SEMANTIC_SIMILARITY_THRESHOLD: float
    SEMANTIC_MAX_TOKENS_PER_CHUNK: int
    SEMANTIC_MIN_TOKENS_PER_CHUNK: int
    CLUSTER_MIN_TOKENS_PER_SENTENCE: int
    CHARACTER_MAX_CHARS: int
    CHARACTER_OVERLAP: int
    PROMPTS_DIR: Path = BASE_DIR / "prompts"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create the input bucket and output bucket names
        self.INPUT_BUCKET_NAME = f"{self.INPUT_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        self.VECTOR_BUCKET_NAME = f"{self.VECTOR_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        self.INPUT_BUCKET_NAME_PDF = f"{self.INPUT_BUCKET_NAME_INIT_PDF}{self.PROJECT_NAME}"
        self.KNOWLEDGEBASE_BUCKET_NAME = (f"{self.KNOWLEDGEBASE_BUCKET_NAME_INIT}{self.PROJECT_NAME}")
        self.FILE_NAME_PDF = f"{Path(self.FILE_NAME).stem}.pdf"
        self._load_credentials()

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".dev_env" if os.getenv("DEV") else BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    def _load_credentials(self):
        if os.getenv("DEV"):
            if self.GOOGLE_APPLICATION_CREDENTIALS is not None:
                try:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_APPLICATION_CREDENTIALS
                except ValueError as e:
                    logger.error(f"error loading development environment variables  - {e}")

settings = Settings()
