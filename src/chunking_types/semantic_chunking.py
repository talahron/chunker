"""
Semantic chunking pipeline based on Cumulative Semantic Chunking.

Flow: start a chunk with sentence #1 -> embed it -> embed sentence #2 ->
if similar enough AND chunk still under the token cap, merge and re-embed
the combined text -> compare to #3 -> keep going until similarity drops
or the chunk hits the token cap, then close and start a new chunk.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings
from src.gcs_manager import GCSManager
from src.pydantic_models import ParagraphData, ParagraphLocation
from src.chunking_types.chunking_utils import ChunkerUtils
from src.agents.caching_agents import (
    DocumentCachingProcessor,
    ContextSentenceSplitAgent,
)
from src.settings import settings
from src.logger import setup_logger

class SemanticChunker(ChunkerUtils):
    """
    End-to-end pipeline: parser JSON -> sentences -> short-sentence filter ->
    cumulative semantic grouping -> enrichment -> indexer output.
    """
    def __init__(self,
                 model_name_sentence_splitter,
                 model_name_enrichment_chunks,
                 thinking_budget_enrichment_chunks,
                 embeddings_model_name,
                 similarity_threshold,
                 max_tokens_per_chunk,
                 min_tokens_per_chunk,
                 project_id,
                 prompts_dir,
                 knowledgebase_bucket_name,
                 file_config,
                 embeddings_region,
                 caching_location,
                 google_provider_region,
                 input_bucket,
                 file_name,
                 increment,
                 max_workers_for_batch_processing,
                 input_bucket_name_pdf=None,
                 file_name_pdf=None,
                 # Passed from ChunkerInitialization but unused here.

                 model_name=None,
                 max_tokens=None,
                 min_tokens=None,
                 cluster_min_tokens_per_sentence=None,
                 character_max_chars=None
                 ) -> None:
        super().__init__()
        self.model_name_sentence_splitter = model_name_sentence_splitter
        self.model_name_enrichment_chunks = model_name_enrichment_chunks
        self.thinking_budget_enrichment_chunks = thinking_budget_enrichment_chunks
        self.embeddings_model_name = embeddings_model_name
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.project_id = project_id
        self.prompts_dir = prompts_dir
        self.knowledgebase_bucket_name = knowledgebase_bucket_name
        self.file_config = file_config
        self.google_provider_region = google_provider_region
        self.caching_location = caching_location
        self.embeddings_region = embeddings_region
        self.input_bucket = input_bucket
        self.file_name = file_name
        self.increment = increment
        self.max_workers_for_batch_processing = max_workers_for_batch_processing
        self.input_bucket_name_pdf = input_bucket_name_pdf
        self.file_name_pdf = file_name_pdf

        self.sentence_agent: Optional[ContextSentenceSplitAgent] = None
        self.embedder: Optional[VertexAIEmbeddings] = None
        self.context_caching_agent: Optional[DocumentCachingProcessor] = None

    def setup(self):
        self.sentence_agent = ContextSentenceSplitAgent(
            model_name=self.model_name_sentence_splitter,
            project_id=self.project_id,
            google_provider_region=self.google_provider_region,
            caching_location=self.caching_location,
            path_to_system_prompt = self.prompts_dir / "context_sentence_split.md",
            path_to_batch_prompt = self.prompts_dir / "context_sentence_batch.md",
            path_to_document_prompt = self.prompts_dir / "context_sentence_document.md",
            max_concurrent=self.max_workers_for_batch_processing,
            bucket_name_pdf=self.input_bucket_name_pdf,
            blob_path_pdf=self.file_name_pdf,
        )

        # Auth flows through ADC (GOOGLE_APPLICATION_CREDENTIALS) - no API key.
        self.embedder = VertexAIEmbeddings(
            model=self.embeddings_model_name,
            project=self.project_id,
            location=self.embeddings_region,
        )

        self.context_caching_agent = DocumentCachingProcessor(
            model_name=self.model_name_enrichment_chunks,
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
                ("general", self.prompts_dir / "knowledgebase_general.md"),
                ("extended", self.prompts_dir / "knowledgebase_extended.md"),
            ],
            document_tagging_agent_prompt_paths=[
                ("document_classification", self.prompts_dir / "document_classification.md"),
                ("evidence_scale", self.prompts_dir / "evidence_scale.md"),
            ],
            bucket_name_pdf=self.input_bucket_name_pdf,
            blob_path_pdf=self.file_name_pdf,
        )

        self.storage = GCSManager()
        self.logger = setup_logger(self.__class__.__name__)
        return self

    def _filter_short_chunks(self, chunks: List[ParagraphData]) -> List[ParagraphData]:
        """Drop chunks whose merged content is below the token threshold."""
        kept = [c for c in chunks if len(c.content) // 4 >= self.min_tokens_per_chunk]
        dropped = len(chunks) - len(kept)
        self.logger.info(
            f"Filtered out {dropped} chunks shorter than {self.min_tokens_per_chunk} tokens "
            f"({len(kept)} remaining)"
        )
        return kept

    async def _embed(self, text: str) -> List[float]:
        """Wrap aembed_query so the caller can await it cleanly."""
        return await self.embedder.aembed_query(text)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two 1-D vectors, in [-1, 1].

        Computed with numpy so the service does not need scikit-learn at
        runtime. Returns 0.0 for a zero-length vector instead of dividing
        by zero.
        """
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    @staticmethod
    def _merge_into_chunk(chunk_sentences: List[ParagraphData],
                          chunk_id: float) -> ParagraphData:
        """Combine a list of sentences into one ParagraphData."""
        file_name = chunk_sentences[0].file_name
        content = " ".join(s.content.strip() for s in chunk_sentences)

        # For each metadata field: build a dict keyed by the dedup value (page number / type string / location int), with the object itself
        # as the value. Duplicate keys naturally overwrite, leaving one entry per unique value in first-seen order. Then we just take .values().
        pages = list({p.page_number: p for s in chunk_sentences for p in s.pages}.values())
        types = list({t.type: t for s in chunk_sentences for t in s.type}.values())

        # build the locations parameter
        nested_list = [par_loc.locations for par_obj in chunk_sentences for par_loc in par_obj.location]
        tuples_mapping = dict([(k, tuple(sublist)) for k, sublist in enumerate(nested_list)])
        # Extract unique values preserving order
        unique_values = list(dict.fromkeys(tuples_mapping.values()))
        # Rebuild dictionary with new sequential keys
        location_values = dict(enumerate(unique_values))
        locations = [ParagraphLocation(locations=sublist) for sublist in location_values.values()]

        return ParagraphData(
            object_id=chunk_id,
            file_name=file_name,
            type=types,
            pages=pages,
            content=content,
            location=locations,
        )

    async def _group_semantically(self, sentences: List[ParagraphData]) -> List[ParagraphData]:
        """
        Walk sentences in order. Start a chunk with the first sentence.
        For each next sentence: embed it, compare to the current chunk's embedding.
        If similar AND chunk still under token cap -> merge and re-embed the
        combined text. Otherwise -> close chunk, start a new one.

        Note: the loop is inherently sequential (each step depends on the
        current chunk state), so the awaits run serially by design.
          * chunks            - the finished chunks produced so far.
                                Each is a merged ParagraphData built by
                                _merge_into_chunk. Grows over time.
          * chunk_sentences   - the sentences accumulated in the CURRENT
                                open chunk. Grows as we merge, resets when
                                we close the chunk and start a new one.
          * chunk_text        - the concatenated text of chunk_sentences.
                                Kept alongside chunk_sentences to avoid
                                re-joining strings on every iteration, and
                                to feed the embedder with one combined text.
          * chunk_embedding   - the embedding vector of chunk_text. This is
                                what we compare each next sentence against.
                                Re-computed every time a sentence joins.
        """
        # filter no values in content - to prevent 400 INVALID_ARGUMENT error in the google.genai.errors.ClientError.
        sentences = [s for s in sentences if s.content and s.content.strip()]
        if not sentences:
            return []

        # Initialize the first (open) chunk from the first sentence:
        # it IS the chunk for now, so its text and its embedding are the
        # current chunk's text and embedding.
        chunks: List[ParagraphData] = []
        chunk_sentences: List[ParagraphData] = [sentences[0]]
        chunk_text: str = sentences[0].content
        chunk_embedding = np.array(await self._embed(chunk_text))

        for idx, sentence in enumerate(sentences[1:], start=2):
            # sentence_embedding - the vector for the candidate sentence on its own.
            # We compare it against the current chunk's embedding to decide fit.
            sentence_embedding = np.array(await self._embed(sentence.content))

            # similarity - cosine similarity in [-1, 1] between the current chunk and the candidate sentence.
            similarity = self._cosine_similarity(chunk_embedding, sentence_embedding)
            # Calculating log variables to understand the current situation
            candidate_text = " ".join([chunk_text, sentence.content])
            candidate_token_count = len(candidate_text) // 4
            current_chunk_tokens = len(chunk_text) // 4
            new_sentence_tokens = len(sentence.content) // 4
            fits_semantically = similarity >= self.similarity_threshold
            fits_size = candidate_token_count <= self.max_tokens_per_chunk
            self.logger.info(
            f"Sentence {idx}/{len(sentences)} | open chunks: {len(chunks) + 1} | "
            f"similarity={similarity:.3f} (threshold={self.similarity_threshold}) | "
            f"tokens: current={current_chunk_tokens} + new={new_sentence_tokens} -> candidate={candidate_token_count} (cap={self.max_tokens_per_chunk}) | "
            f"decision={'MERGE' if fits_semantically and fits_size else 'SPLIT'}"
            )

            if similarity >= self.similarity_threshold and candidate_token_count <= self.max_tokens_per_chunk:
                # Merge into the current chunk and re-embed the combined text
                # so the next comparison reflects the full accumulated context.
                chunk_sentences.append(sentence)
                chunk_text = candidate_text
                chunk_embedding = np.array(await self._embed(chunk_text))
            else:
                # Close the current chunk, start a new one from this sentence.
                chunks.append(self._merge_into_chunk(chunk_sentences, float(len(chunks) + 1)))
                chunk_sentences = [sentence]
                chunk_text = sentence.content
                chunk_embedding = sentence_embedding

        # Finalize the last open chunk.
        chunks.append(self._merge_into_chunk(chunk_sentences, float(len(chunks) + 1)))

        self.logger.info(
            f"Cumulative semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks"
        )
        return chunks

    async def pipeline(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.logger.info("SemanticChunker pipeline: START")

        # 1) Download parser JSON.
        await self._download_parser_json()
        if not self.document:
            self.logger.warning("Document is empty - returning empty output")
            return [], []

        # 2) Split into sentences.
        self.logger.info("Sentence splitting: START")
        sentences = await self._split_into_sentences()
        self.logger.info(f"Sentence splitting: END ({len(sentences)} sentences)")
        if not sentences:
            return [], []

        # 3) Cumulative semantic grouping.
        self.logger.info("Cumulative semantic grouping: START")
        semantic_chunks = await self._group_semantically(sentences)
        self.logger.info(f"Cumulative semantic grouping: END ({len(semantic_chunks)} chunks)")

        # 4) Drop chunks that are too short to be useful for retrieval.
        semantic_chunks = self._filter_short_chunks(semantic_chunks)
        if not semantic_chunks:
            self.logger.warning("No chunks left after short-chunk filter")
            return [], []

        # 5) Enrich each chunk with PDF context.
        self.logger.info("Context caching enrichment: START")
        enriched_chunks = await self.context_caching_agent.enrich_chunks_pipeline(
            chunks=semantic_chunks
        )
        self.logger.info("Context caching enrichment: END")

        # 6) Build (contents, metadata) for the indexer.
        self.logger.info("Build content and metadata: START")
        contents, metadata = await self._build_content_and_metadata(enriched_chunks)
        self.logger.info("Build content and metadata: END")

        self.logger.info(f"SemanticChunker pipeline: DONE ({len(contents)} chunks)")

        return contents, metadata
