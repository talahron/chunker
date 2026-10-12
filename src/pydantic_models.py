from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
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
class DocumentSubject(str, Enum):
    CRIMINAL_LAW = "משפט פלילי"
    CIVIL_LAW = "משפט אזרחי"
    ADMINISTRATIVE_LAW = "משפט מנהלי"
    LABOR_LAW = "דיני עבודה"
    OTHER = "אחר"

class DocumentType(str, Enum):
    VERDICT = "פסק דין"
    DECISION = "החלטה"
    CONTRACT = "חוזה"
    PROCEDURE = "נוהל"
    CORRESPONDENCE = "התכתבות"
    OTHER = "אחר"

class DocumentSubjectTag(BaseModel):
    """LLM response schema for the document subject tagging agent"""
    document_subject: DocumentSubject = Field(..., description="The main subject of the document")

class DocumentTypeTag(BaseModel):
    """LLM response schema for the document type tagging agent"""
    document_type: DocumentType = Field(..., description="The type of the document")

class DocumentTags(BaseModel):
    """All tags of one document. A field stays None when its tagging agent failed."""
    # frozen - one instance is shared by every chunk of the document
    # extra="forbid" - every tagging response field must have a matching field here
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_subject: Optional[DocumentSubject] = None
    document_type: Optional[DocumentType] = None

class ChunkMetaData(DocumentTags):
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
    # Set by DocumentCachingProcessor, never by the LLM - hidden from the agentic output schema.
    document_tags: SkipJsonSchema[DocumentTags] = Field(default_factory=DocumentTags,
                                                        description="Tags of the source document")

class AgenticResponse(BaseModel):
    """The expected response format from using agentic chunking"""
    paragraphs: list[ParagraphData] = Field(
        ...,
        description="List of aggregated paragraphs in order"
    )

class ChunkingStrategy(str, Enum):
    HEADER_CHUNKING = "header"
    AGENTIC_CHUNKING = "agentic"
    SEMANTIC_CHUNKING = "semantic"
    CLUSTER_SEMANTIC_CHUNKING = "cluster"
    CHARACTER_CHUNKING = "character"

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
