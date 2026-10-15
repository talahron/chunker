from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

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

# --- Document tags: document-level labels spread onto every chunk ---
# Placeholder values - replace with the real lists.
class DocumentClassification(str, Enum):
    CRIMINAL_LAW = "משפט פלילי"
    CIVIL_LAW = "משפט אזרחי"
    ADMINISTRATIVE_LAW = "משפט מנהלי"
    LABOR_LAW = "דיני עבודה"
    OTHER = "אחר"

class EvidenceScale(str, Enum):
    VERDICT = "פסק דין"
    DECISION = "החלטה"
    CONTRACT = "חוזה"
    PROCEDURE = "נוהל"
    CORRESPONDENCE = "התכתבות"
    OTHER = "אחר"

class DocumentClassificationTag(BaseModel):
    """LLM response schema for the document classification tagging agent"""
    document_classification: DocumentClassification = Field(..., description="The classification of the document")

class EvidenceScaleTag(BaseModel):
    """LLM response schema for the evidence scale tagging agent"""
    evidence_scale: EvidenceScale = Field(..., description="The evidence scale of the document")

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
