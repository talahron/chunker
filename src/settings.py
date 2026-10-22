import os
from pathlib import Path
from typing import ClassVar, Literal, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GCP_PROJECT_ID: str
    GCP_LOCATION: str
    WORKER_MODE: bool = False
    WORKER_STAGE: Literal["chunker"] = "chunker"
    WORKER_SUBSCRIPTION: str = ""
    WORKER_CHILD_COMMAND: str = "main.py"
    WORKER_MAX_MESSAGES: int = Field(default=1, ge=1)
    WORKER_MAX_LEASE_SECONDS: int = Field(default=43200, ge=1)
    WORKER_SHUTDOWN_GRACE_SECONDS: int = Field(default=900, ge=0)
    CACHING_LOCATION: str
    EMBEDDINGS_REGION: str
    INPUT_BUCKET_NAME_INIT: str
    INPUT_BUCKET_NAME_INIT_PDF: str
    INPUT_BUCKET_NAME: Optional[str] = None
    INPUT_BUCKET_NAME_PDF: Optional[str] = None
    PROJECT_NAME: str = ""
    FILE_NAME: str = ""
    FILE_NAME_PDF: Optional[str] = None
    MODEL_NAME: str
    MODEL_NAME_ENRICHMENT_CHUNKER: str
    THINKING_BUDGET_ENRICHMENT_CHUNKER: int
    MAX_TOKENS: int
    MIN_TOKENS: int
    EMBEDDINGS_MODEL_NAME: str
    INDEX_REGION: str
    VECTOR_BUCKET_NAME_INIT: str
    VECTOR_BUCKET_NAME: Optional[str] = None
    KNOWLEDGEBASE_BUCKET_NAME_INIT: str
    KNOWLEDGEBASE_BUCKET_NAME: str = ""
    # The tenant whose files get a knowledge base per case and the tagging agents (tnufa in production)
    CASE_TENANT_NAME: str = "qwen"
    INDEX_MAPPING_BUCKET: str = "moj-gcp-d-rag-index-mapping-bucket"
    INDEX_MAPPING_OBJECT_PREFIX: str = "indexes"
    FILE_METADATA_BUCKET_NAME_INIT: str = "moj-gcp-d-rag-input-"
    FILE_METADATA_BUCKET_NAME: Optional[str] = None
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

    WORKER_SOURCE_BUCKET_ENV: ClassVar[str] = "INPUT_BUCKET_NAME"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create the input bucket and output bucket names
        self.INPUT_BUCKET_NAME = self.INPUT_BUCKET_NAME or f"{self.INPUT_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        self.VECTOR_BUCKET_NAME = self.VECTOR_BUCKET_NAME or f"{self.VECTOR_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        self.INPUT_BUCKET_NAME_PDF = (
            self.INPUT_BUCKET_NAME_PDF
            or f"{self.INPUT_BUCKET_NAME_INIT_PDF}{self.PROJECT_NAME}"
        )
        self.KNOWLEDGEBASE_BUCKET_NAME = (
            self.KNOWLEDGEBASE_BUCKET_NAME
            or f"{self.KNOWLEDGEBASE_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        )
        self.FILE_METADATA_BUCKET_NAME = (
            self.FILE_METADATA_BUCKET_NAME
            or f"{self.FILE_METADATA_BUCKET_NAME_INIT}{self.PROJECT_NAME}"
        )
        self.FILE_NAME_PDF = self.FILE_NAME_PDF or f"{Path(self.FILE_NAME).stem}.pdf"
        self._load_credentials()

    model_config = SettingsConfigDict(
        env_file=(
            (BASE_DIR / ".env_dev", BASE_DIR / ".dev_env")
            if os.environ.get("DEV")
            else BASE_DIR / ".env"
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    def _load_credentials(self):
        if os.environ.get("DEV"):
            if self.GOOGLE_APPLICATION_CREDENTIALS is not None:
                try:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_APPLICATION_CREDENTIALS
                except ValueError as e:
                    logger.error(f"error loading development environment variables  - {e}")

    @model_validator(mode="after")
    def validate_worker_configuration(self):
        if self.WORKER_MODE and not self.WORKER_SUBSCRIPTION.strip():
            raise ValueError("WORKER_SUBSCRIPTION is required when WORKER_MODE is true")
        return self

settings = Settings()
