import asyncio
from src.exceptions import ChunkingDataError, IngestToVectorError
from src.settings import settings
from src.pydantic_models import ChunkingStrategy, FileConfig
from src.gcs_manager import GCSManager

if settings.WORKER_MODE:
    from worker import PubSubListener

    pub_sub_listener = PubSubListener(
        project_id=settings.GCP_PROJECT_ID,
        bucket=settings.INPUT_BUCKET_NAME_INIT,
        worker_stage=settings.WORKER_STAGE,
        source_bucket_env=settings.WORKER_SOURCE_BUCKET_ENV,
        worker_subscription=settings.WORKER_SUBSCRIPTION,
        worker_child_command=settings.WORKER_CHILD_COMMAND,
        worker_max_messages=settings.WORKER_MAX_MESSAGES,
        worker_max_lease_seconds=settings.WORKER_MAX_LEASE_SECONDS,
        worker_shutdown_grace_seconds=settings.WORKER_SHUTDOWN_GRACE_SECONDS,
    ).setup()

    raise SystemExit(pub_sub_listener.run_worker())

from src.pydantic_models import ChunkingStrategy
from src.chunking_types.header_chunking import HeaderChunker
from src.chunking_types.agentic_chunking import AgenticChunker
from src.chunking_types.semantic_chunking import SemanticChunker
from src.chunking_types.character_chunking import CharacterChunker
from src.chunking_types.cluster_semantic_chunking import ClusterSemanticChunker
from src.Inserts_data_into_vector_db import Indexer  # keep your existing module name

from src.logger import setup_logger, configure_logging
from src import http_trace

configure_logging()
http_trace.install()
logger = setup_logger(__name__)

class ChunkerInitialization:
    # ── construction ──
    def __init__(self, file_config: FileConfig):
        self.file_config = file_config
        self.model_name = settings.MODEL_NAME
        self.model_name_enrichment_chunks = settings.MODEL_NAME_ENRICHMENT_CHUNKER
        self.model_name_sentence_splitter = settings.MODEL_NAME_SENTENCE_SPLITTER
        self.thinking_budget_enrichment_chunks = settings.THINKING_BUDGET_ENRICHMENT_CHUNKER
        self.embeddings_model_name = settings.EMBEDDINGS_MODEL_NAME
        self.project_id = settings.GCP_PROJECT_ID
        self.google_provider_region = settings.GCP_LOCATION
        self.embeddings_region = settings.EMBEDDINGS_REGION
        self.caching_location = settings.CACHING_LOCATION
        self.prompts_dir = settings.PROMPTS_DIR

        # Buckets and file names
        self.knowledgebase_bucket_name = settings.KNOWLEDGEBASE_BUCKET_NAME
        self.input_bucket = settings.INPUT_BUCKET_NAME
        self.file_name = settings.FILE_NAME
        self.input_bucket_name_pdf = settings.INPUT_BUCKET_NAME_PDF
        self.file_name_pdf = settings.FILE_NAME_PDF

        # Chunking params
        self.max_tokens = settings.MAX_TOKENS
        self.min_tokens = settings.MIN_TOKENS
        self.increment = settings.INCREMENT
        self.chunking_strategy = file_config.CHUNKING_TYPE
        self.max_workers_for_batch_processing = settings.MAX_WORKERS_FOR_BATCH_PROCESSING

        # Semantic chunker params
        self.similarity_threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD
        self.max_tokens_per_chunk = settings.SEMANTIC_MAX_TOKENS_PER_CHUNK
        self.min_tokens_per_chunk = settings.SEMANTIC_MIN_TOKENS_PER_CHUNK

        # Cluster semantic chunker params
        self.cluster_min_tokens_per_sentence = settings.CLUSTER_MIN_TOKENS_PER_SENTENCE

        # Character chunker params
        self.character_max_chars = settings.CHARACTER_MAX_CHARS

        # Map strategies to their corresponding chunker classes
        self.chunker_map = {
            ChunkingStrategy.HEADER_CHUNKING: HeaderChunker,
            ChunkingStrategy.AGENTIC_CHUNKING: AgenticChunker,
            ChunkingStrategy.SEMANTIC_CHUNKING: SemanticChunker,
            ChunkingStrategy.CLUSTER_SEMANTIC_CHUNKING: ClusterSemanticChunker,
            ChunkingStrategy.CHARACTER_CHUNKING: CharacterChunker,
        }

    def setup(self):
        """Initialize and setup the appropriate chunker based on strategy."""
        chunker_class = self.chunker_map.get(self.chunking_strategy)

        if chunker_class is None:
            raise ValueError(
                f"Unknown chunking strategy: {self.chunking_strategy}. "
                f"Valid options are: {list(self.chunker_map.keys())}"
            )

        # Instantiate the selected chunker with common parameters
        chunker = chunker_class(
            model_name=self.model_name,
            model_name_enrichment_chunks=self.model_name_enrichment_chunks,
            model_name_sentence_splitter=self.model_name_sentence_splitter,
            thinking_budget_enrichment_chunks=self.thinking_budget_enrichment_chunks,
            embeddings_model_name=self.embeddings_model_name,
            similarity_threshold=self.similarity_threshold,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
            min_tokens_per_chunk=self.min_tokens_per_chunk,
            cluster_min_tokens_per_sentence=self.cluster_min_tokens_per_sentence,
            character_max_chars=self.character_max_chars,
            project_id=self.project_id,
            prompts_dir=self.prompts_dir,
            knowledgebase_bucket_name=self.knowledgebase_bucket_name,
            file_config=self.file_config,
            google_provider_region=self.google_provider_region,
            embeddings_region=self.embeddings_region,
            caching_location=self.caching_location,
            input_bucket=self.input_bucket,
            file_name=self.file_name,
            input_bucket_name_pdf=self.input_bucket_name_pdf,
            file_name_pdf=self.file_name_pdf,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            increment=self.increment,
            max_workers_for_batch_processing=self.max_workers_for_batch_processing
        ).setup()

        return chunker

class IndexerInitialization:
    # ── construction ──
    def __init__(self, texts_for_index, metadata_for_index, file_config: FileConfig):
        self.texts_for_index = texts_for_index
        self.metadata_for_index = metadata_for_index
        self.index_parameters = file_config.index_parameters

        # Index / Embeddings
        self.batch_size = settings.BATCH_SIZE
        self.embeddings_model_name = settings.EMBEDDINGS_MODEL_NAME
        self.project_name = settings.PROJECT_NAME
        self.vector_bucket= settings.VECTOR_BUCKET_NAME
        self.index_region = settings.INDEX_REGION

    def setup(self):
        indexer = Indexer(text=self.texts_for_index,
                          meta_data=self.metadata_for_index,
                          batch_size=self.batch_size,
                          index_parameters=self.index_parameters,
                          embeddings_model_name=self.embeddings_model_name,
                          project_name=self.project_name,
                          vector_bucket=self.vector_bucket,
                          index_region=self.index_region,
                          ).setup()
        return indexer

async def main():
    """
    Orchestrate the full flow from GCS JSON -> chunking -> metadata.
    """
    logger = setup_logger("main")

    file_config = await GCSManager().load_file_config(
        file_name=settings.FILE_NAME,
        project_name=settings.PROJECT_NAME,
        index_mapping_bucket=settings.INDEX_MAPPING_BUCKET,
        index_mapping_object_prefix=settings.INDEX_MAPPING_OBJECT_PREFIX,
        file_metadata_bucket_name=settings.FILE_METADATA_BUCKET_NAME,
        is_case_tenant=settings.PROJECT_NAME == settings.CASE_TENANT_NAME,
    )

    try:
        # Build the chunker over the raw document list (it sorts internally)
        chunker = ChunkerInitialization(file_config).setup()

        # Creates the chunks ready for the vector database
        texts_for_index, metadata_for_index = await chunker.pipeline()
        logger.info(f"✅ Finished chunking {len(texts_for_index)} chunks")
    except ChunkingDataError as e:
        logger.error(f'Failed to Chunk the input json')
        raise
    try:
        # Ingest to vector index
        indexer = IndexerInitialization(texts_for_index, metadata_for_index, file_config).setup()
        await indexer.ingest_data()
        logger.info(f"✅ Finished indexing {len(texts_for_index)} chunks")
    except IngestToVectorError as e:
        logger.error(f'Failed to ingest the Chunk to vector index')
        raise

if __name__ == "__main__":
    asyncio.run(main())
