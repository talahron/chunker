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
    """The route of one processed file: its project's index-mapping JSON + its own metadata JSON"""
    # indexes/{PROJECT_NAME}.json
    INDEX_END_POINT_ID: str
    INDEX_ID: str
    PROJECT_ID: str
    PROJECT_NUMBER: str
    CUSTOMER: str  # kept from the JSON; the route is decided by uses_tnufa_route
    EXTRACTION_MODEL: ExtractionModel
    CHUNKING_TYPE: ChunkingStrategy
    PYDANTIC_MODEL: Union[dict[str, Any], str]
    EXTENDED_KNOWLEDGEBASE: bool
    # Not from that JSON
    uses_tnufa_route: bool  # PROJECT_NAME == TNUFA_TENANT_NAME: knowledge base per case and tagging agents
    file_metadata: dict[str, Any]  # {FILE}.json after it passed PYDANTIC_MODEL

    def runs_agent(self, agent_name: str) -> bool:
        """Whether this file runs the given document agent (knowledge base or tagging)"""
        switches = {
            "general": True,  # the general knowledge base always runs
            "extended": self.EXTENDED_KNOWLEDGEBASE,
            # Tagging agents run only on the TNUFA route, by the flags in the file metadata
            "document_classification": self.uses_tnufa_route and self.file_metadata.get("DOCUMENT_CLASSIFICATION") is True,
            "evidence_scale": self.uses_tnufa_route and self.file_metadata.get("EVIDENCE_SCALE") is True,
        }
        return switches[agent_name]

    def knowledgebase_file_name(self, agent_name: str) -> str:
        """general_<CASE_NUMBER>.md on the TNUFA route when the file has a case number, otherwise general.md"""
        case_number = self.file_metadata.get("CASE_NUMBER") if self.uses_tnufa_route else None
        if case_number:
            return f"{agent_name}_{case_number}.md"
        return f"{agent_name}.md"

    def document_metadata(self, tags: dict[str, Any]) -> dict[str, Any]:
        """The document-level fields that every chunk of this file gets in the index.

        Called once per file in STEP 5 of enrich_chunks_pipeline, after the tagging agents ran.
        _build_content_and_metadata then adds the result to the metadata of every chunk.

        Regular route: the fields of the file metadata JSON, as they are.
        TNUFA route: the DOCUMENT_CLASSIFICATION / EVIDENCE_SCALE flags only decided which
        tagging agents run, so they are dropped and the tags those agents returned take their place.
        Example: {"DOCUMENT_CLASSIFICATION": true, "DISTRICT": "NORTH", "CASE_NUMBER": "518", "EVIDENCE_SCALE": true}
        + tags {"document_classification": "צווים", "evidence_scale": "החלטה"}
        -> {"DISTRICT": "NORTH", "CASE_NUMBER": "518", "document_classification": "צווים", "evidence_scale": "החלטה"}
        """
        metadata = dict(self.file_metadata)
        if self.uses_tnufa_route:
            # The flags only decided which tagging agents run, so the index gets the tags instead of the flags
            metadata.pop("DOCUMENT_CLASSIFICATION", None)
            metadata.pop("EVIDENCE_SCALE", None)
            metadata.update(tags)
        return metadata

    @property
    def index_parameters(self) -> dict[str, str]:
        """The keys the Indexer used to read from the index mapping JSON"""
        return {
            "INDEX_ID": self.INDEX_ID,
            "INDEX_END_POINT_ID": self.INDEX_END_POINT_ID,
            "PROJECT_NUMBER": self.PROJECT_NUMBER,
        }

# JSON Schema type -> Python type, for the flat schema in PYDANTIC_MODEL
JSON_SCHEMA_TYPES = {"string": str, "boolean": bool, "integer": int, "number": float, "array": list, "object": dict}

def create_model_from_json_schema(schema: dict[str, Any] | str, model_name: str = "FileMetadata") -> type[BaseModel]:
    """Build the model that a file metadata JSON must match: every field required, no extra fields, exact types"""
    if isinstance(schema, str):
        schema = json.loads(schema)
    fields = {name: (JSON_SCHEMA_TYPES[field["type"]], ...) for name, field in schema["properties"].items()}
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
