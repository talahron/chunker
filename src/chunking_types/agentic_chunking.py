from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from src.gcs_manager import GCSManager
from src.exceptions import SummarizerAgentError, BatchPredictionError
from src.pydantic_models import DocumentObject, Chunk, ChunkMetaData, ObjectMetaData, ParagraphData
import asyncio
import numpy as np
from src.chunking_types.chunking_utils import ChunkerUtils
from src.agents.caching_agents import DocumentCachingProcessor
from src.agents.regular_agents import AgenticTextAgent
from src.logger import setup_logger


class AgenticChunker(ChunkerUtils):
    """Utility class creating agentic chunking of the parser json output."""

    # ── construction ──
    def __init__(self,
                 model_name,
                 model_name_enrichment_chunks,
                 thinking_budget_enrichment_chunks,
                 project_id,
                 prompts_dir,
                 knowledgebase_bucket_name,
                 file_config,
                 google_provider_region,
                 embeddings_region,
                 caching_location,
                 input_bucket,
                 file_name,
                 input_bucket_name_pdf,
                 file_name_pdf,
                 max_workers_for_batch_processing,
                 max_tokens: Optional[int],
                 min_tokens:Optional[int],
                 increment: Optional[int],
                 # Params accepted from ChunkerInitialization but unused by this chunker.
                 model_name_sentence_splitter: Optional[Any] = None,
                 embeddings_model_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 max_tokens_per_chunk: Optional[int] = None,
                 min_tokens_per_chunk: Optional[int] = None,
                 cluster_min_tokens_per_sentence: Optional[int] = None,
                 character_max_chars: Optional[int] = None,
                 character_overlap: Optional[int] = None,
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_name_enrichment_chunks = model_name_enrichment_chunks
        self.thinking_budget_enrichment_chunks = thinking_budget_enrichment_chunks
        self.project_id = project_id
        self.prompts_dir = prompts_dir
        self.knowledgebase_bucket_name = knowledgebase_bucket_name
        self.file_config = file_config
        self.google_provider_region = google_provider_region
        self.embeddings_region = embeddings_region
        self.caching_location = caching_location
        self.input_bucket = input_bucket
        self.file_name = file_name
        self.input_bucket_name_pdf = input_bucket_name_pdf
        self.file_name_pdf = file_name_pdf
        self.max_workers_for_batch_processing = max_workers_for_batch_processing
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.increment = increment
        self.document: list[DocumentObject] = []
        self.agentic_agent = None
        self.context_caching_agent = None
        self.storage = None
        self.logger = None

    def setup(self):
        self.agentic_agent = AgenticTextAgent(model_name=self.model_name,
                                              project_id=self.project_id,
                                              google_provider_region=self.google_provider_region,
                                              path_to_system_prompt=self.prompts_dir / "agentic_text.md",)

        self.agentic_batch_prompt = (self.prompts_dir / "agentic_batch.md").read_text(encoding="utf-8")

        self.context_caching_agent = DocumentCachingProcessor(model_name=self.model_name_enrichment_chunks,
                                                         thinking_budget_enrichment_chunks=self.thinking_budget_enrichment_chunks,
                                                         project_id=self.project_id,
                                                         google_provider_region=self.google_provider_region,
                                                         caching_location=self.caching_location,
                                                         knowledgebase_bucket_name=self.knowledgebase_bucket_name,
                                                         file_config=self.file_config,
                                                         path_to_enrichment_prompt=(
                                                                      self.prompts_dir / "document_context_enrichment.md"
                                                              ),
                                                        knowledgebase_agent_prompt_paths=[
                                                                  ("general",
                                                                   self.prompts_dir / "knowledgebase_general.md"),
                                                                  ("extended",
                                                                   self.prompts_dir / "knowledgebase_extended.md"),
                                                              ],
                                                              document_tagging_agent_prompt_paths=[
                                                                  ("document_classification",
                                                                   self.prompts_dir / "document_classification.md"),
                                                                  ("evidence_scale",
                                                                   self.prompts_dir / "evidence_scale.md"),
                                                              ],
                                                              bucket_name_pdf=self.input_bucket_name_pdf,
                                                              blob_path_pdf=self.file_name_pdf)
        self.storage = GCSManager()
        self.logger = setup_logger(self.__class__.__name__)
        return self


    async def create_agentic_paragraphs(self) -> List[ParagraphData]:
        """creating agentic paragraphs based on the complete parser json object"""
        # convert the self.document into list[dict[str,str]] - {"object_id":..., "file_name":..., "type":..., "page":..., "location":..., "content":...}
        converted_documents = self._convert_parser_output_to_dict
        # create a batch of texts
        batch_texts = self._batch_prediction(converted_documents)
        agentic_paragraph = []

        semaphore = asyncio.Semaphore(self.max_workers_for_batch_processing)

        async def process_single_batch(k, batch):
            async with semaphore:
                self.logger.debug(f"Running the agentic chunking process on batch number - {k} / {len(batch_texts)}")
                try:
                    prompt = self.agentic_batch_prompt.format(batch=batch)
                    batch_paragraphs = await self.agentic_agent.run(prompt)
                    self.logger.info("Successfully created agentic paragraphs")
                    return batch_paragraphs.paragraphs
                except Exception as e:
                    self.logger.error(f"failed to process batch - {k} / {len(batch_texts)} - {e}")
                    return []

        tasks = [
            process_single_batch(k, batch)
            for k, batch in enumerate(batch_texts)
        ]
        results = await asyncio.gather(*tasks)

        for output in results:
            agentic_paragraph.extend(output)

        # Every batch numbers its paragraphs from 1.0, so renumber them in document order
        for number, paragraph in enumerate(agentic_paragraph, start=1):
            paragraph.object_id = float(number)

        self.logger.info(
            f"Done! Successfully created agentic paragraphs"
        )
        return agentic_paragraph



    async def pipeline(self)-> Tuple[List[str], List[ChunkMetaData]]:
        """
        Orchestrate the full flow from GCS JSON -> chunking -> metadata.
        """
        self.logger.info("Create chunks: start")

        # Downloading the initial parser object from the GCS bucket
        self.logger.info(f"Downloading parser json object - {self.file_name}")
        await self._download_parser_json()
        self.logger.info(f'Successfully download parser json. Document name - {self.file_name}')

        # create the agentic paragraphs
        self.logger.info("Create agentic paragraphs: START")
        agentic_paragraphs = await self.create_agentic_paragraphs()
        self.logger.info("Create agentic paragraphs: END")

        # add context caching to the agentic paragraphs
        self.logger.info("Add context caching to the agentic paragraphs: START")
        agentic_paragraphs = await self.context_caching_agent.enrich_chunks_pipeline(chunks=agentic_paragraphs)
        self.logger.info("Add context caching to the agentic paragraphs: END")

        # build the content and metadata
        self.logger.info("Build content and metadata: START")
        contents, metadata = await self._build_content_and_metadata(agentic_paragraphs)
        self.logger.info("Build content and metadata: END")

        return contents, metadata
