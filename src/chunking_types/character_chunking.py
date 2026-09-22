"""
Character Chunker.

Walks the parser DocumentObjects in order and concatenates their content into
fixed-size chunks of `character_max_chars` characters. If a single object's
content is larger than the cap, it is sliced mid-text (the size cap is hard).

No overlap: contextual continuity is provided downstream by the enrichment
step (DocumentCachingProcessor) which has access to the full PDF.

Each emitted chunk preserves the metadata (pages, types, locations) of every
DocumentObject that contributed any text to it.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import re

from src.gcs_manager import GCSManager
from src.pydantic_models import (DocumentObject, ParagraphData, ParagraphType,
    ParagraphPageNumber, ParagraphLocation,)
from src.chunking_types.chunking_utils import ChunkerUtils
from src.agents.caching_agents import DocumentCachingProcessor
from src.logger import setup_logger


class CharacterChunker(ChunkerUtils):
    """
    End-to-end pipeline: parser JSON -> fixed-size character chunks
    -> enrichment -> indexer output.
    """

    def __init__(self,
                 model_name_enrichment_chunks,
                 thinking_budget_enrichment_chunks,
                 character_max_chars,
                 project_id,
                 prompts_dir,
                 knowledgebase_bucket_name,
                 google_provider_region,
                 embeddings_region,
                 caching_location,
                 input_bucket,
                 file_name,
                 increment,
                 max_workers_for_batch_processing,
                 input_bucket_name_pdf=None,
                 file_name_pdf=None,
                 # Passed from ChunkerInitialization but unused here.
                 model_name=None,
                 model_name_sentence_splitter=None,
                 embeddings_model_name=None,
                 similarity_threshold=None,
                 max_tokens_per_chunk=None,
                 min_tokens_per_chunk=None,
                 cluster_min_tokens_per_sentence=None,
                 max_tokens=None,
                 min_tokens=None,

                 ) -> None:
        super().__init__()
        self.model_name_enrichment_chunks = model_name_enrichment_chunks
        self.thinking_budget_enrichment_chunks = thinking_budget_enrichment_chunks
        self.character_max_chars = character_max_chars
        self.project_id = project_id
        self.prompts_dir = prompts_dir
        self.knowledgebase_bucket_name = knowledgebase_bucket_name
        self.google_provider_region = google_provider_region
        self.caching_location = caching_location
        self.input_bucket = input_bucket
        self.file_name = file_name
        self.increment = increment
        self.max_workers_for_batch_processing = max_workers_for_batch_processing
        self.input_bucket_name_pdf = input_bucket_name_pdf
        self.file_name_pdf = file_name_pdf

        self.context_caching_agent: Optional[DocumentCachingProcessor] = None
        self.embeddings_region = embeddings_region

    def setup(self):
        self.context_caching_agent = DocumentCachingProcessor(
            model_name=self.model_name_enrichment_chunks,
            thinking_budget_enrichment_chunks=self.thinking_budget_enrichment_chunks,
            project_id=self.project_id,
            google_provider_region=self.google_provider_region,
            caching_location=self.caching_location,
            knowledgebase_bucket_name=self.knowledgebase_bucket_name,
            path_to_enrichment_prompt=(
                    self.prompts_dir / "document_context_enrichment.md"
            ),
            knowledgebase_agent_prompt_paths=[
                ("general", self.prompts_dir / "knowledgebase_general.md"),
                ("extended", self.prompts_dir / "knowledgebase_extended.md"),
            ],
            bucket_name_pdf=self.input_bucket_name_pdf,
            blob_path_pdf=self.file_name_pdf,
        )
        self.storage = GCSManager()
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(f"CharacterChunker configured: max_chars={self.character_max_chars}")
        return self

    @staticmethod
    def _build_chunk(content: str,
                     contributing_objects: List[DocumentObject],
                     chunk_id: float) -> ParagraphData:
        """
        Wrap a chunk's text and the source DocumentObjects that contributed
        to it into a ParagraphData. Pages/types/locations are deduplicated
        while preserving first-seen order.
        """
        pages = list({obj.page: ParagraphPageNumber(page_number=obj.page)
                      for obj in contributing_objects}.values())
        types = list({obj.type: ParagraphType(type=obj.type)
                      for obj in contributing_objects}.values())

        # build the locations parameter
        nested_list = [obj.location for obj in contributing_objects]
        tuples_mapping = dict([(k, tuple(sublist)) for k, sublist in enumerate(nested_list)])
        # Extract unique values preserving order
        unique_values = list(dict.fromkeys(tuples_mapping.values()))
        # Rebuild dictionary with new sequential keys
        location_values = dict(enumerate(unique_values))
        locations = [ParagraphLocation(locations=sublist) for sublist in location_values.values()]

        return ParagraphData(
            object_id=chunk_id,
            file_name=contributing_objects[0].file_name,
            type=types,
            pages=pages,
            content=content,
            location=locations,
        )

    def _split_by_characters(self, document: List[DocumentObject]) -> List[ParagraphData]:
        """
        Walk the parser DocumentObjects in order, accumulating text until the
        chunk reaches max_chars, then emit it and start a new one. An object
        bigger than max_chars is sliced into multiple chunks mid-text.

        State:
          * chunks         - finished ParagraphData chunks.
          * buffer         - text accumulated for the chunk currently being built.
          * buffer_objects - DocumentObjects that contributed any text to buffer.
        """
        if not document:
            return []

        chunks: List[ParagraphData] = []
        buffer: str = ""
        buffer_objects: List[DocumentObject] = []

        for obj in document:
            text = re.sub(r"\s+", " ", obj.content).strip()

            # Skip empty objects so they don't pollute the chunk's metadata
            # without contributing any actual text.
            if not text:
                continue

            # Prepend a space when joining a new object's text to a non-empty buffer so words from different objects don't get glued together.
            # if buffer:
            #     text += " "

            # Append text from this object to the buffer, slicing across as many
            # chunks as needed if the text doesn't fit. Each iteration of the
            # while-loop fills one chunk's worth (or what's left of the text).
            while text:
                # Calculates each time how much space is left in the buffer until it needs to be closed.
                space_left = self.character_max_chars - len(buffer)
                # Adds the currently checked text into the buffer.
                buffer = " ".join([buffer,text[:space_left]]).strip()
                # Adds the entire text object currently being examined into the buffer.
                buffer_objects.append(obj)
                # Removes from the text currently being reviewed what was taken to the buffer.
                text = text[space_left:]

                # Buffer hit the character_max_chars: emit and reset.
                if len(buffer) >= self.character_max_chars:
                    chunks.append(self._build_chunk(buffer, buffer_objects, float(len(chunks) + 1)))
                    self.logger.info(f"Emitted chunk {len(chunks)} | length={len(buffer)} chars")
                    buffer = ""
                    buffer_objects = []

        # Flush whatever's left as the final chunk.
        if buffer:
            chunks.append(self._build_chunk(buffer, buffer_objects, float(len(chunks) + 1)))
            self.logger.info(f"Emitted final chunk {len(chunks)} | length={len(buffer)} chars")

        self.logger.info(f"Character chunking: {len(document)} objects -> {len(chunks)} chunks")
        return chunks

    async def pipeline(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.logger.info("CharacterChunker pipeline: START")

        # 1) Download parser JSON.
        await self._download_parser_json()
        if not self.document:
            self.logger.warning("Document is empty - returning empty output")
            return [], []

        # 2) Character-based chunking over the parser content.
        self.logger.info("Character-based chunking: START")
        character_chunks = self._split_by_characters(self.document)
        self.logger.info(f"Character-based chunking: END ({len(character_chunks)} chunks)")
        if not character_chunks:
            return [], []

        # 3) Enrich each chunk with PDF context.
        self.logger.info("Context caching enrichment: START")
        enriched_chunks = await self.context_caching_agent.enrich_chunks_pipeline(
            chunks=character_chunks
        )
        self.logger.info("Context caching enrichment: END")

        # 4) Build (contents, metadata) for the indexer.
        self.logger.info("Build content and metadata: START")
        contents, metadata = await self._build_content_and_metadata(enriched_chunks)
        self.logger.info("Build content and metadata: END")

        self.logger.info(f"CharacterChunker pipeline: DONE ({len(contents)} chunks)")
        return contents, metadata
