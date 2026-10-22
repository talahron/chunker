import json
from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, create_model
from enum import Enum
from typing import List, Literal, Optional, Union

# --- Document tags: document-level labels spread onto every chunk ---
class DocumentClassification(str, Enum):
    TESTIMONIES_AND_INTERROGATIONS = "עדויות וחקירות באזהרה"
    POLICE_ACTIONS_SUBSTANTIVE_INVESTIGATION = "פעולות של המשטרה - חקירה מהותית"
    POLICE_ACTIONS_EVIDENCE_ANALYSIS_SUMMARY = "פעולות של המשטרה - סיכום ניתוח ראיות"
    POLICE_ACTIONS_TECHNICAL = "פעולות של המשטרה - טכניות בתיק"
    EVIDENCE_SEIZED_PHYSICAL_EXHIBITS = "ראיות - מוצגים פיזיים שנתפסו"
    EVIDENCE_BANKING_INFO = "ראיות - מידע בנקאי"
    EVIDENCE_ACCOUNTING_INFO = "ראיות - מידע חשבונאי"
    WARRANTS_AND_ORDERS = "צווים"
    COURT_MOTIONS_PROTOCOLS_DECISIONS = "בקשות, פרוטוקולים והחלטות בית משפט"
    PUBLIC_SERVANT_CERTIFICATES = "תעודות עובד ציבור ומערכות ציבוריות"
    INSTITUTIONAL_RECORDS = "רשומות מוסדיות"
    EXPERT_OPINIONS = "חוות דעת מומחה"
    WIRETAPPING = "האזנות סתר"
    DIGITAL_FORENSICS_MOBILE = "פורנזיקה דיגיטלית - הפקות פלאפונים"
    DIGITAL_FORENSICS_COMPUTERS_EMAILS_MEDIA = "פורנזיקה דיגיטלית - הפקות מיילים / קבצי מחשב / מדיה"

class EvidenceScale(str, Enum):
    VERDICT = "פסק דין"
    DECISION = "החלטה"
    CONTRACT = "חוזה"
    PROCEDURE = "נוהל"
    CORRESPONDENCE = "התכתבות"
    OTHER = "אחר"

class ChunkingStrategy(str, Enum):
    HEADER_CHUNKING = "HEADER"
    AGENTIC_CHUNKING = "AGENTIC"
    SEMANTIC_CHUNKING = "SEMANTIC"
    CLUSTER_SEMANTIC_CHUNKING = "CLUSTER"
    CHARACTER_CHUNKING = "CHARACTER"

class ExtractionModel(str, Enum):
    GEMINI = "GEMINI"
    QWEN = "QWEN"

class FileConfig(BaseModel):
    """The route of one processed file - read from the JSON that sits next to the file"""
    INDEX_END_POINT_ID: str
    INDEX_ID: str
    PROJECT_ID: str
    PROJECT_NUMBER: str
    CUSTOMER: str
    DISTRICT: str
    CASE_NUMBER: str = ""
    EXTRACTION_MODEL: ExtractionModel
    CHUNKING_TYPE: ChunkingStrategy
    PYDANTIC_MODEL: Optional[str] = ""
    DOCUMENT_CLASSIFICATION: bool
    EVIDENCE_SCALE: bool
    EXTENDED_KNOWLEDGEBASE: bool

    def runs_agent(self, agent_name: str) -> bool:
        """Whether this file runs the given document agent (knowledge base or tagging)"""
        switches = {
            "general": True,  # the general knowledge base always runs
            "extended": self.EXTENDED_KNOWLEDGEBASE,
            "document_classification": self.DOCUMENT_CLASSIFICATION,
            "evidence_scale": self.EVIDENCE_SCALE,
        }
        return switches[agent_name]

    def knowledgebase_file_name(self, agent_name: str) -> str:
        """general.md, or general_<CASE_NUMBER>.md when the file belongs to a case"""
        if self.CASE_NUMBER:
            return f"{agent_name}_{self.CASE_NUMBER}.md"
        return f"{agent_name}.md"

    @property
    def index_parameters(self) -> dict[str, str]:
        """The keys the Indexer used to read from the index mapping JSON"""
        return {
            "INDEX_ID": self.INDEX_ID,
            "INDEX_END_POINT_ID": self.INDEX_END_POINT_ID,
            "PROJECT_NUMBER": self.PROJECT_NUMBER,
        }

# JSON Schema types allowed in PYDANTIC_MODEL; the schema must be flat
JSON_SCHEMA_TYPES = {"string": str, "boolean": bool, "integer": int, "number": float, "array": list, "object": dict}
ALLOWED_PROPERTY_KEYS = {"type", "items", "description", "title"}

def create_model_from_json_schema(schema: dict[str, Any] | str, model_name: str = "FileMetadata") -> type[BaseModel]:
    """Build the model that a file metadata JSON must match: every field required, no extra fields, exact types"""
    if isinstance(schema, str):
        schema = json.loads(schema)
    if schema.get("type", "object") != "object":
        raise ValueError(f"PYDANTIC_MODEL must describe an object, got type {schema.get('type')!r}")

    fields = {}
    for name, field_schema in schema.get("properties", {}).items():
        unsupported = set(field_schema) - ALLOWED_PROPERTY_KEYS
        if unsupported:
            raise ValueError(f"PYDANTIC_MODEL field {name!r} uses unsupported keys: {sorted(unsupported)}")
        field_type = JSON_SCHEMA_TYPES.get(field_schema.get("type"))
        if field_type is None:
            raise ValueError(f"PYDANTIC_MODEL field {name!r} has unsupported type {field_schema.get('type')!r}")
        if field_type is list and "items" in field_schema:
            item_type = JSON_SCHEMA_TYPES.get(field_schema["items"].get("type"))
            if item_type in (None, list, dict):
                raise ValueError(f"PYDANTIC_MODEL field {name!r} must hold a list of simple values")
            field_type = list[item_type]
        fields[name] = (field_type, ...)

    return create_model(model_name, __config__=ConfigDict(extra="forbid", strict=True), **fields)

class DocumentClassificationTag(BaseModel):
    """LLM response schema for the document classification tagging agent"""
    document_classification: DocumentClassification = Field(..., description="The classification of the document")

class EvidenceScaleTag(BaseModel):
    """LLM response schema for the evidence scale tagging agent"""
    evidence_scale: EvidenceScale = Field(..., description="The evidence scale of the document")

class DocumentObject(BaseModel):
    object_id: str
    file_name: str
    type: str
    page: int
    location: List[int] = Field(description="The page object bounding box [y_min, x_min, y_max, x_max] positions on the page, normalized to a 0–1000 scale",
                                default_factory=list)
    content: str
    chunk_number: Optional[float] = None  # set later by header-based pass

class ObjectMetaData(BaseModel):
    object_id: str
    file_name: str
    type: str
    page: int
    location: List[int] = Field(description="The page object bounding box [y_min, x_min, y_max, x_max] positions on the page, normalized to a 0–1000 scale",
                                default_factory=list)

class Chunk(BaseModel):
    chunk_number: float = Field(..., description="Chunk ID", examples=[1.0, 2.0, 3.0])
    content: str = ""
    meta_data: List[ObjectMetaData] = []
    @property
    def token_count(self) -> int:
        # ~4 chars per token (Hebrew heuristic)
        return len(self.content) // 4

class ChunkMetaData(BaseModel):
    file_name: str
    chunk_number: float
    pages: str
    document_classification: Optional[DocumentClassification] = None
    evidence_scale: Optional[EvidenceScale] = None

class SubChunk(BaseModel):
    """Single semantic segment returned from the LLM."""
    paragraph: str = Field(..., description="assigned a unique key to each paragraph")
    content: str = Field(..., description="content of the paragraph or section")

class SubChunks(BaseModel):
    sub_chunks: Optional[List[SubChunk]]

class ParagraphType(BaseModel):
    type: str = Field(..., description="The type of the content", examples=["text", "Image", "list_item", "section_header", "Table"])

class ParagraphPageNumber(BaseModel):
    page_number: int = Field(..., description="The page number where the content was found")

class ParagraphLocation(BaseModel):
    locations: List[int] = Field(description="The page object bounding box [y_min, x_min, y_max, x_max] positions on the page, normalized to a 0–1000 scale",
                                 default_factory=list)

class ParagraphData(BaseModel):
    """aggregated text into paragraph"""
    object_id: float = Field(..., description="A unique identifier for the aggregated paragraph", examples=[1.0, 2.0, 3.0])
    file_name: str = Field(..., description="Name of the source file")
    type: list[ParagraphType] = Field(..., description="Type of the paragraph")
    pages: list[ParagraphPageNumber] = Field(..., description="Aggregated page numbers from which the paragraph's content was sourced")
    content: str = Field(..., description="Aggregated text of a paragraph")
    location: list[ParagraphLocation] = Field(..., description="The positions of the contents on the page")
    # Document-level tags, set on every chunk by DocumentCachingProcessor
    document_classification: Optional[DocumentClassification] = Field(default=None, description="The classification of the source document")
    evidence_scale: Optional[EvidenceScale] = Field(default=None, description="The evidence scale of the source document")


class AgenticResponse(BaseModel):
    """The expected response format from using agentic chunking"""
    paragraphs: list[ParagraphData] = Field(
        ...,
        description="List of aggregated paragraphs in order"
    )


# --- LLM DTO Models ---
class SimpleParagraphData(BaseModel):
    object_id: float
    file_name: str
    type: list[str]
    pages: list[int]
    content: str
    locations: List[int] = Field(description="The page object bounding box [y_min, x_min, y_max, x_max] positions on the page, normalized to a 0–1000 scale",
                                 default_factory=list)

class SimpleAgenticResponse(BaseModel):
    paragraphs: list[SimpleParagraphData]
