"""
Cluster Semantic Chunker (Chroma Research, "Evaluating Chunking Strategies").

Algorithm:
  1. Embed every sentence (already split upstream).
  2. Build an NxN similarity matrix of dot products between embeddings.
  3. Subtract the matrix mean - values above-average become positive (=good),
     below-average become negative (=bad). Zero out the diagonal so a sentence
     can't earn reward by being similar to itself.
  4. Dynamic programming finds the partition of sentences into contiguous
     groups that maximizes the total reward (sum of pairwise similarities
     within each group), subject to a max group size.
  5. Each group becomes a chunk.

Reference:
  https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py

Difference from SemanticChunker (Cumulative):
  * Cumulative is greedy and local - walks sentence by sentence.
  * Cluster is global - sees the full document and finds the optimal
    partition under the size constraint.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings
from src.gcs_manager import GCSManager
from src.pydantic_models import ParagraphData, ParagraphLocation, DocumentSubjectTag, DocumentTypeTag
from src.chunking_types.chunking_utils import ChunkerUtils
from src.agents.caching_agents import (
    DocumentCachingProcessor,
    ContextSentenceSplitAgent,
)

from src.settings import settings
from src.logger import setup_logger


class ClusterSemanticChunker(ChunkerUtils):
    """
    End-to-end pipeline: parser JSON -> sentences -> cluster-based semantic
    grouping -> short-chunk filter -> enrichment -> indexer output.
    """

    def __init__(self,
                 model_name_sentence_splitter,
                 model_name_enrichment_chunks,
                 thinking_budget_enrichment_chunks,
                 embeddings_model_name,
                 max_tokens_per_chunk,
                 min_tokens_per_chunk,
                 cluster_min_tokens_per_sentence,
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
                 max_tokens=None,
                 min_tokens=None,
                 similarity_threshold=None,
                 character_max_chars=None,
                 ) -> None:
        super().__init__()
        self.model_name_sentence_splitter = model_name_sentence_splitter
        self.model_name_enrichment_chunks = model_name_enrichment_chunks
        self.thinking_budget_enrichment_chunks = thinking_budget_enrichment_chunks
        self.embeddings_model_name = embeddings_model_name
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        # Average sentence size in tokens - used to derive max_cluster_size
        # (max sentences allowed in one chunk). Mirrors the Chroma reference,
        # which divides max_chunk_size by min_chunk_size.
        self.cluster_min_tokens_per_sentence = cluster_min_tokens_per_sentence
        self.project_id = project_id
        self.prompts_dir = prompts_dir
        self.knowledgebase_bucket_name = knowledgebase_bucket_name
        self.google_provider_region = google_provider_region
        self.embeddings_region = embeddings_region
        self.caching_location = caching_location
        self.input_bucket = input_bucket
        self.file_name = file_name
        self.increment = increment
        self.max_workers_for_batch_processing = max_workers_for_batch_processing
        self.input_bucket_name_pdf = input_bucket_name_pdf
        self.file_name_pdf = file_name_pdf

        self.sentence_agent: Optional[ContextSentenceSplitAgent] = None
        self.embedder: Optional[VertexAIEmbeddings] = None
        self.context_caching_agent: Optional[DocumentCachingProcessor] = None
        # max_cluster_size derived in setup() once we have the params.
        self.max_cluster_size: int = 0

    def setup(self):
        self.sentence_agent = ContextSentenceSplitAgent(
            model_name=self.model_name_sentence_splitter,
            project_id=self.project_id,
            google_provider_region=self.google_provider_region,
            caching_location=self.caching_location,
            path_to_system_prompt=self.prompts_dir / "context_sentence_split.md",
            path_to_batch_prompt=self.prompts_dir / "context_sentence_batch.md",
            path_to_document_prompt=self.prompts_dir / "context_sentence_document.md",
            max_concurrent=self.max_workers_for_batch_processing,
            bucket_name_pdf=self.input_bucket_name_pdf,
            blob_path_pdf=self.file_name_pdf,
        )

        # Auth flows through ADC (GOOGLE_APPLICATION_CREDENTIALS) - no API key.
        self.embedder = VertexAIEmbeddings(
            model=self.embeddings_model_name,
            project=self.project_id,
            location=self.embeddings_region
        )

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

        # max_cluster_size = how many sentences can pack into one chunk before
        # exceeding max_tokens_per_chunk. Same idea as Chroma:
        # max_chunk_size // min_sentence_size.
        self.max_cluster_size = max(1, self.max_tokens_per_chunk // self.cluster_min_tokens_per_sentence)

        self.storage = GCSManager()
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            f"ClusterSemanticChunker configured: max_cluster_size={self.max_cluster_size} "
            f"(={self.max_tokens_per_chunk} tokens / {self.cluster_min_tokens_per_sentence} per sentence)"
        )
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

    async def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in one go and return them as an (n, dim) numpy array."""
        vectors = await self.embedder.aembed_documents(texts[:250])
        return np.array(vectors)

    @staticmethod
    def _merge_into_chunk(chunk_sentences: List[ParagraphData], chunk_id: float) -> ParagraphData:
        """Combine a list of sentences into one ParagraphData."""
        file_name = chunk_sentences[0].file_name
        content = " ".join(s.content.strip() for s in chunk_sentences)

        # For each metadata field: build a dict keyed by the dedup value (page number / type string / location int), with the object itself
        # as the value. Duplicate keys naturally overwrite, leaving one entry per unique value in first-seen order. Then we just take .values().
        pages = list({p.page_number: p for s in chunk_sentences for p in s.pages}.values())
        types = list({t.type: t for s in chunk_sentences for t in s.type}.values())

        # build the locations parameter
        nested_list = [loc.locations for m in chunk_sentences for loc in m.location]
        tuples_mapping = dict([(k, tuple(sublist)) for k, sublist in enumerate(nested_list)])
        # Extract unique values preserving order
        unique_values = list(dict.fromkeys(tuples_mapping.values()))
        # Rebuild dictionary with new sequential keys
        location_values = dict(enumerate(unique_values))
        locations = [ParagraphLocation(locations=sublist) for sublist in location_values.values()]

        # locations = list({loc.location: loc for s in chunk_sentences for loc in s.location}.values())

        return ParagraphData(
            object_id=chunk_id,
            file_name=file_name,
            type=types,
            pages=pages,
            content=content,
            location=locations,
        )

    @staticmethod
    def _build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Pairwise dot product of every embedding with every other. Output shape: (n, n)."""
        return np.dot(embeddings, embeddings.T)

    @staticmethod
    def _normalize_matrix_for_dp(matrix: np.ndarray) -> np.ndarray:
        """
        Center the matrix around zero by subtracting the mean of its
        upper-triangle (the unique pairs, excluding the diagonal). After
        centering, above-average pairs are positive rewards and below-average
        pairs are negative penalties. The diagonal is zeroed so a sentence's
        self-similarity never inflates a group's score.
        """
        upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
        mean_value = float(np.mean(upper_triangle))

        normalized = matrix - mean_value
        np.fill_diagonal(normalized, 0)
        return normalized

    @staticmethod
    def _calculate_reward(matrix: np.ndarray, start: int, end: int) -> float:
        """
        Reward for grouping sentences[start..end] (inclusive) into one chunk:
        the sum of all pairwise similarities (post-normalization) within
        that index range. Higher = more internally cohesive group.
        """
        sub_matrix = matrix[start:end + 1, start:end + 1]
        return float(np.sum(sub_matrix))

    def _optimal_segmentation(self, matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Dynamic programming over the normalized similarity matrix.
        Returns a list of (start_index, end_index) pairs, inclusive on both
        ends, partitioning [0..n-1] contiguously with no gaps.

        DP state:
          dp[i] = best total reward for partitioning sentences 0..i.
          segmentation[i] = the start index of the LAST group in that best
                            partition (so we can backtrack to recover it).

        Transition:
          For each position i, try every group size from 1..max_cluster_size
          ending at i. The best total is the group's own reward plus the best
          reward of everything before the group started.
        """
        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, self.max_cluster_size + 1):
                start = i - size + 1
                if start < 0:
                    break  # group would extend past the start of the array

                reward = self._calculate_reward(matrix, start, i)
                total = reward + (dp[start - 1] if start > 0 else 0.0)

                if total > dp[i]:
                    dp[i] = total
                    segmentation[i] = start

        # Walk segmentation backwards to recover the actual groups.
        clusters: List[Tuple[int, int]] = []
        i = n - 1
        while i >= 0:
            start = int(segmentation[i])
            clusters.append((start, i))
            i = start - 1
        clusters.reverse()
        return clusters

    async def _group_by_clustering(self, sentences: List[ParagraphData]) -> List[ParagraphData]:
        """
        Embed all sentences, run the DP segmentation, and turn each cluster
        into a single ParagraphData via _merge_into_chunk.
        """
        # filter no values in content - to prevent 400 INVALID_ARGUMENT error in the google.genai.errors.ClientError.
        sentences = [s for s in sentences if s.content and s.content.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._merge_into_chunk(sentences, chunk_id=1.0)]

        # Embed every sentence at once. Cluster needs the full matrix up front,
        # unlike Cumulative which embeds incrementally.
        self.logger.info(f"Embedding {len(sentences)} sentences (batch)")
        texts = [s.content for s in sentences]
        embeddings = await self._embed_batch(texts)

        self.logger.info("Building similarity matrix")
        similarity_matrix = self._build_similarity_matrix(embeddings)

        self.logger.info("Normalizing matrix and running DP segmentation")
        normalized = self._normalize_matrix_for_dp(similarity_matrix)
        clusters = self._optimal_segmentation(normalized)

        self.logger.info(f"DP produced {len(clusters)} clusters")

        # Build a ParagraphData per cluster. chunk_id is 1-based sequential.
        chunks: List[ParagraphData] = []
        for k, (start, end) in enumerate(clusters, start=1):
            chunk_sentences = sentences[start:end + 1]
            chunks.append(self._merge_into_chunk(chunk_sentences, chunk_id=float(k)))
            self.logger.info(
                f"Cluster {k}/{len(clusters)}: sentences[{start}..{end}] "
                f"({end - start + 1} sentences merged)"
            )
        return chunks

    async def pipeline(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.logger.info("ClusterSemanticChunker pipeline: START")

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

        # 3) Cluster-based semantic grouping (DP over similarity matrix).
        self.logger.info("Cluster semantic grouping: START")
        semantic_chunks = await self._group_by_clustering(sentences)
        self.logger.info(f"Cluster semantic grouping: END ({len(semantic_chunks)} chunks)")

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

        self.logger.info(f"ClusterSemanticChunker pipeline: DONE ({len(contents)} chunks)")

        return contents, metadata
