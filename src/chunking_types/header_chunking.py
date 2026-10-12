from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

from src.gcs_manager import GCSManager
from src.exceptions import PipelineTaskExecutorError, LargeChunksSplitAgentError
from src.pydantic_models import (
    DocumentObject, Chunk, ObjectMetaData, ParagraphData,
    ParagraphType, ParagraphPageNumber, ParagraphLocation,
    DocumentSubjectTag, DocumentTypeTag,
)
from src.agents.caching_agents import DocumentCachingProcessor
from src.agents.regular_agents import LargeChunksSplitAgent
from src.chunking_types.chunking_utils import ChunkerUtils
from src.logger import setup_logger


class HeaderChunker(ChunkerUtils):
    """
    Header-based chunker.

    Pipeline: parser JSON -> assign chunk numbers by section_header -> aggregate
    objects per chunk -> split oversized chunks via LLM -> filter tiny chunks
    -> convert to ParagraphData -> enrichment -> indexer output.

    Output is aligned with all other chunkers (ParagraphData) so the rest of
    the pipeline (enrichment, _build_content_and_metadata) is shared.
    """

    # ── construction ──
    def __init__(self,
                 model_name,
                 model_name_enrichment_chunks,
                 thinking_budget_enrichment_chunks,
                 project_id,
                 prompts_dir,
                 knowledgebase_bucket_name,
                 google_provider_region,
                 embeddings_region,
                 caching_location,
                 input_bucket,
                 file_name,
                 max_tokens: Optional[int],
                 min_tokens: Optional[int],
                 increment: Optional[int],
                 input_bucket_name_pdf=None,
                 file_name_pdf=None,
                 # Params accepted from ChunkerInitialization but unused by this chunker.
                 model_name_sentence_splitter: Optional[Any] = None,
                 max_workers_for_batch_processing: Optional[int] = None,
                 embeddings_model_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 max_tokens_per_chunk: Optional[int] = None,
                 min_tokens_per_chunk: Optional[int] = None,
                 cluster_min_tokens_per_sentence: Optional[int] = None,
                 character_max_chars: Optional[int] = None,
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_name_enrichment_chunks = model_name_enrichment_chunks
        self.thinking_budget_enrichment_chunks = thinking_budget_enrichment_chunks
        self.project_id = project_id
        self.prompts_dir = prompts_dir
        self.knowledgebase_bucket_name = knowledgebase_bucket_name
        self.google_provider_region = google_provider_region
        self.input_bucket = input_bucket
        self.file_name = file_name
        self.input_bucket_name_pdf = input_bucket_name_pdf
        self.file_name_pdf = file_name_pdf
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.increment = increment

        self.chunks: Dict[float, Chunk] = {}
        self.document: List[DocumentObject] = []
        self.LargeChunksSplitAgent = None
        self.context_caching_agent: Optional[DocumentCachingProcessor] = None
        self.storage = None
        self.logger = None
        self.embeddings_region = embeddings_region
        self.caching_location = caching_location

    def setup(self):
        self.LargeChunksSplitAgent = LargeChunksSplitAgent(model_name=self.model_name,
                                                           project_id=self.project_id,
                                                           google_provider_region=self.google_provider_region,
                                                           path_to_system_prompt=self.prompts_dir / "large_chunks_split.md",)
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
            document_tagging_agent_prompt_paths=[
                (DocumentSubjectTag, self.prompts_dir / "document_tagging_subject.md"),
                (DocumentTypeTag, self.prompts_dir / "document_tagging_type.md"),
            ],
            bucket_name_pdf=self.input_bucket_name_pdf,
            blob_path_pdf=self.file_name_pdf,
        )
        self.storage = GCSManager()
        self.logger = setup_logger(self.__class__.__name__)
        return self

    # ───────────────────────── header-based chunking ──▶
    def _header_based_chunks(self):
        """Assign sequential chunk IDs whenever a `section_header` is encountered."""
        current_chunk = 0
        # make sure that the self.document is not an empty list
        if not self.document:
            self.logger.debug(f"self.document is an empty list - stopped executing the function for file name - {self.file_name}")
            return []
        else:
            for obj in self.document:
                if obj.type == "section_header":
                    current_chunk += 1
                obj.chunk_number = float(current_chunk)
            self.logger.info(f"Inserting chunk numbers based on headers completed successfully. number of objects: {len(self.document)}")
            return None

    def _aggregate_chunks(self) -> None:
        """Aggregate all document objects into chunks by their chunk_number."""

        # temporary dictionary to hold chunks, keyed by chunk_number
        chunks_dict: Dict[float, Chunk] = {}

        # make sure that the self.document is not an empty list
        if not self.document:
            self.logger.debug("self.document object is an empty list - stopped executing the function")
            return None

        # Iterate over all document objects
        for k, document_object in enumerate(self.document):
            self.logger.info(f"running aggregation process on object number - {k} / {len(self.document)}")
            # Saves the chunk number
            chunk_id = document_object.chunk_number

            # If the chunk does not exist yet, create a new Chunk
            if chunk_id not in chunks_dict:
                self.logger.info(f"chunk does not exist yet - creating new chunk for object number - {k} / {len(self.document)}")
                chunks_dict[chunk_id] = Chunk(
                    chunk_number=chunk_id,
                    content=document_object.content,
                    meta_data=[],
                )

            # If the chunk already exists, Append the content:
            else:
                current_content = chunks_dict[chunk_id].content
                new_content = document_object.content
                chunks_dict[chunk_id].content = " ".join([current_content, new_content])

            # Append the metadata of the current document object into the chunk
            chunks_dict[chunk_id].meta_data.append(
                ObjectMetaData(
                    object_id=document_object.object_id,
                    file_name=document_object.file_name,
                    type=document_object.type,
                    page=document_object.page,
                    location=document_object.location,
                )
            )
            self.logger.info(f"Successfully running the aggregated chunk process on object number - {k} / {len(self.document)}")

        # Save all chunks sorted by their chunk_number into self.chunks
        self.chunks = {k: chunks_dict[k] for k in sorted(chunks_dict.keys())}
        self.logger.info(f"Aggregation complete successfully. Number of chunks: {len(self.chunks)}")
        return None

    def _filter_small_chunks(self, *, min_tokens: int = 50) -> None:
        """Filter in-place: keep only chunks with enough tokens."""
        self.chunks = {
            k: c for k, c in self.chunks.items()
            if c.token_count >= min_tokens
        }
        # Keep ordering by key
        self.chunks = {k: self.chunks[k] for k in sorted(self.chunks.keys())}
        self.logger.info(f"Filter small chunks complete successfully. Number of chunks: {len(self.chunks)}")

    # ───────────────────────── split large chunks ──▶
    async def _split_large_chunks(self, *, max_tokens: int = 500) -> None:
        """Replace every chunk whose `token_count` exceeds `max_tokens` with decimal-indexed sub-chunks."""
        new_chunks: Dict[float, Chunk] = {}
        # Iterate in key order to preserve deterministic behavior
        for k in sorted(self.chunks.keys()):
            chunk = self.chunks[k]
            # Checks if the current chunk has  tokens amount < max_tokens - continues to the next chunk
            if chunk.token_count <= max_tokens:
                new_chunks[chunk.chunk_number] = chunk
                continue

            # Hold the current chunk number
            base_id = int(float(chunk.chunk_number))

            try:
                self.logger.info(f"running large chunk splitter on chunk number - {chunk.chunk_number} / {len(self.chunks)}")
                # Returns the list of sub-chunks from the agent
                sub_chunks: List[str] = await self.LargeChunksSplitAgent.run(chunk.content)
                self.logger.info(f"Successfully running large chunk splitter on chunk number - {chunk.chunk_number} / {len(self.chunks)}")
            except LargeChunksSplitAgentError as e:
                self.logger.error(f"Split agent failed; keeping original chunk. chunk_number: {chunk.chunk_number} / {len(self.chunks)} - {e}")
                new_chunks[chunk.chunk_number] = chunk
                continue

            # If the agent returned nothing, the original large chunk remains.
            if not sub_chunks:
                self.logger.warning(f"""Large chunks split agent returned no paragraphs;
                                        keeping original chunk_number: {chunk.chunk_number}.
                                        token_count: {chunk.token_count}""")
                new_chunks[chunk.chunk_number] = chunk
                continue

            # Iterating the list of subchunks returned from the agent
            for idx, para in enumerate(sub_chunks, start=1):
                # Keeps the new chunk number - sub_id_str
                sub_id_str = f"{base_id}.{idx:03d}"  # Work assumption!! There is a limit of up to 999 sub-chunks
                sub_key = float(sub_id_str)
                # Inserts the sub-chunks into the new list
                new_chunks[sub_key] = Chunk(
                    chunk_number=sub_key,
                    content=para,
                    meta_data=chunk.meta_data,  # keep provenance
                )
        # Overwrites all the values in self.chunks to the new list (new_chunks_list) that includes the subchunks.
        self.chunks = {k: new_chunks[k] for k in sorted(new_chunks.keys())}
        self.logger.info(f"Split large chunks complete successfully.. Number of chunks: {len(self.chunks)}")

    # ───────────────────────── conversion to common ParagraphData format ──▶
    def _convert_chunks_to_paragraph_data(self) -> List[ParagraphData]:
        """
        Convert internal Chunk objects into the common ParagraphData format
        used by all other chunkers, so the rest of the pipeline (enrichment,
        _build_content_and_metadata) can consume them directly.

        For each Chunk, every ObjectMetaData becomes a contribution to the
        chunk's pages/types/locations lists. Duplicates are dropped while
        keeping first-seen order, exactly like the other chunkers do.
        """
        paragraphs: List[ParagraphData] = []
        for chunk_number in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_number]
            file_name = chunk.meta_data[0].file_name if chunk.meta_data else ""

            # For each metadata field: dedup-by-key dict, then take .values().
            pages = list({
                             m.page: ParagraphPageNumber(page_number=m.page)
                             for m in chunk.meta_data
                         }.values())
            types = list({
                             m.type: ParagraphType(type=m.type)
                             for m in chunk.meta_data
                         }.values())
            # build the locations parameter
            nested_list = [m.location for m in chunk.meta_data]
            tuples_mapping = dict([(k, tuple(sublist)) for k, sublist in enumerate(nested_list)])
            # Extract unique values preserving order
            unique_values = list(dict.fromkeys(tuples_mapping.values()))
            # Rebuild dictionary with new sequential keys
            location_values = dict(enumerate(unique_values))
            locations = [ParagraphLocation(locations=sublist) for sublist in location_values.values()]

            paragraphs.append(ParagraphData(
                object_id=chunk.chunk_number,
                file_name=file_name,
                type=types,
                pages=pages,
                content=chunk.content or "",
                location=locations,
            ))
        self.logger.info(f"Converted {len(paragraphs)} chunks to ParagraphData")
        return paragraphs

    # ───────────────────────── pipeline ──▶
    async def pipeline(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Orchestrate the full flow from GCS JSON -> chunking -> enrichment -> output.
        """
        try:
            self.logger.info("HeaderChunker pipeline: START")

            # 1) Download parser JSON from GCS.
            self.logger.info(f"Downloading parser json object - {self.file_name}")
            await self._download_parser_json()
            self.logger.info(f'Successfully download parser json. Document name - {self.file_name}')
            if not self.document:
                self.logger.warning("Document is empty - returning empty output")
                return [], []

            # 2) Assign header-based chunk IDs.
            self.logger.info("Assign header-based chunk IDs: START")
            self._header_based_chunks()
            self.logger.info("Assign header-based chunk IDs: END")

            # 3) Aggregate by chunk id.
            self.logger.info("Aggregate by chunk id: START")
            self._aggregate_chunks()
            self.logger.info("Aggregate by chunk id: END")

            # 4) Split oversized chunks via LLM.
            self.logger.info("Split oversized chunks: START")
            await self._split_large_chunks(max_tokens=self.max_tokens)
            self.logger.info("Split oversized chunks: END")

            # 5) Filter tiny chunks.
            self.logger.info("Filter tiny chunks: START")
            self._filter_small_chunks(min_tokens=self.min_tokens)
            self.logger.info("Filter tiny chunks: END")
            if not self.chunks:
                self.logger.warning("No chunks left after filtering - returning empty output")
                return [], []

            # 6) Convert to common ParagraphData format used by all other chunkers.
            self.logger.info("Convert to ParagraphData: START")
            header_chunks = self._convert_chunks_to_paragraph_data()
            self.logger.info("Convert to ParagraphData: END")

            # 7) Enrich each chunk with PDF context (same as the other chunkers).
            self.logger.info("Context caching enrichment: START")
            enriched_chunks = await self.context_caching_agent.enrich_chunks_pipeline(
                chunks=header_chunks
            )
            self.logger.info("Context caching enrichment: END")

            # 8) Build (contents, metadata) for the indexer (inherited from ChunkerUtils).
            self.logger.info("Build content and metadata: START")
            contents, metadata = await self._build_content_and_metadata(enriched_chunks)
            self.logger.info("Build content and metadata: END")

            self.logger.info(f"HeaderChunker pipeline: DONE ({len(contents)} chunks)")
            return contents, metadata

        except PipelineTaskExecutorError as e:
            self.logger.debug(f"Pipeline execution failed for file name - {self.file_name} - {e}")
            return [], []
