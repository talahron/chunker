"""Per-file route config: which index, knowledge base files, chunking type and agents a file gets."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from src.logger import setup_logger
from src.pydantic_models import ChunkingStrategy

logger = setup_logger(__name__)

# Temporary stand-in for the bucket: every file gets this config until load_file_config reads from GCS.
DUMMY_FILE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "dummy_file_config.json"


class FileConfig(BaseModel):
    """The JSON that sits next to every processed file. Every route decision is made here."""
    model_config = ConfigDict(frozen=True)

    INDEX_END_POINT_ID: str
    INDEX_ID: str
    PROJECT_ID: str
    PROJECT_NUMBER: str
    CUSTOMER: str
    DISTRICT: str
    CASE_NUMBER: str = ""
    EXTRACTION_MODEL: Literal["GEMINI", "QWEN"]
    CHUNKING_TYPE: ChunkingStrategy
    DOCUMENT_CLASSIFICATION: bool
    EVIDENCE_SCALE: bool
    EXTENDED_KNOWLEDGEBASE: bool

    @field_validator("INDEX_END_POINT_ID", "INDEX_ID", "PROJECT_NUMBER", mode="before")
    @classmethod
    def keep_ids_exact(cls, value):
        # A JSON float holds ~16 digits, so a 19-digit ID written as 5.79e+18 can silently become another ID.
        if isinstance(value, float):
            raise ValueError("write IDs as strings (or integers) - a float loses digits")
        value = str(value).strip()
        if not (value.isascii() and value.isdigit()):
            raise ValueError(f"an ID must contain only digits, got {value!r}")
        return value

    @field_validator("CHUNKING_TYPE", mode="before")
    @classmethod
    def match_chunking_strategy(cls, value):
        # "SEMANTIC" in the JSON -> ChunkingStrategy.SEMANTIC_CHUNKING ("semantic")
        return value.strip().lower() if isinstance(value, str) else value

    def runs_agent(self, agent_name: str) -> bool:
        """Whether this file runs the given document agent (knowledge base or tagging)."""
        switches = {
            "general": True,  # the general knowledge base always runs
            "extended": self.EXTENDED_KNOWLEDGEBASE,
            "document_classification": self.DOCUMENT_CLASSIFICATION,
            "evidence_scale": self.EVIDENCE_SCALE,
        }
        return switches[agent_name]  # KeyError = a document agent with no switch in the file config

    def knowledgebase_file_name(self, agent_name: str) -> str:
        """general.md, or general_<CASE_NUMBER>.md when the customer keeps a knowledge base per case (TNUFA)."""
        if self.CASE_NUMBER:
            return f"{agent_name}_{self.CASE_NUMBER}.md"
        return f"{agent_name}.md"

    @property
    def index_parameters(self) -> dict[str, str]:
        """The keys the Indexer used to read from the index mapping JSON."""
        return {
            "INDEX_ID": self.INDEX_ID,
            "INDEX_END_POINT_ID": self.INDEX_END_POINT_ID,
            "PROJECT_NUMBER": self.PROJECT_NUMBER,
        }


async def load_file_config(file_name: str) -> FileConfig:
    """Load the route config of the processed file."""
    if not file_name:
        raise ValueError("FILE_NAME is empty - there is no file config to load")
    config_name = Path(file_name).with_suffix(".json").as_posix()

    # TODO: replace these two lines with the JSON named after the file in its bucket, e.g.:
    #   data, _ = await GCSManager().download_json(bucket_name, config_name)
    logger.info(f"Using dummy file config {DUMMY_FILE_CONFIG_PATH.name} instead of {config_name}")
    data = json.loads(DUMMY_FILE_CONFIG_PATH.read_text(encoding="utf-8"))

    file_config = FileConfig.model_validate(data)
    logger.info(f"File config for {file_name}: {file_config}")
    return file_config
