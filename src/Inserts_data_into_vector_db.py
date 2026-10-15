import json
from hashlib import sha256

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import aiplatform, storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VectorSearchVectorStore

from src.exceptions import DownloadJsonError
from src.logger import setup_logger


class Indexer:
    required_index_parameters = {
        "INDEX_ID",
        "PROJECT_NUMBER",
        "INDEX_END_POINT_ID",
    }

    def __init__(self,
                 text,
                 meta_data,
                 batch_size,
                 index_mapping_bucket,
                 index_mapping_json_file,
                 index_mapping_object_prefix,
                 embeddings_model_name,
                 project_name,
                 vector_bucket,
                 index_region,
                 ) -> None:
        # Input data and configuration
        self.text = text
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.embeddings_model_name = embeddings_model_name
        self.project_name = project_name

        self.index_mapping_bucket = index_mapping_bucket
        self.index_mapping_json_file = index_mapping_json_file
        self.index_mapping_object_prefix = index_mapping_object_prefix

        # vector search index initialization parameters
        self.vector_bucket = vector_bucket
        self.bucket_uri = f"gs://{vector_bucket}"
        self.index_region = index_region
        self.index_id = None  # index_id
        self.project_id = None  # project_id
        self.index_endpoint_id = None  # index_endpoint_id

        # Internal objects initialized later
        self.embeddings = None
        self.my_index = None
        self.my_index_endpoint = None
        self.logger = None

        # set-up logger
        self.logger = setup_logger(self.__class__.__name__)

    def _post_init_(self):
        self.logger.info(
            "Downloading index parameters for project - %s", self.project_name
        )
        index_parameters = self._load_index_parameters_from_json_file()
        self.index_id = str(index_parameters["INDEX_ID"])
        self.project_id = str(index_parameters["PROJECT_NUMBER"])
        self.index_endpoint_id = str(index_parameters["INDEX_END_POINT_ID"])
        self.logger.info(
            "Successfully downloaded index parameters for project - %s",
            self.project_name,
        )

    def _download_json_file(self, object_name: str):
        client = storage.Client()
        bucket = client.bucket(self.index_mapping_bucket)
        content = bucket.blob(object_name).download_as_bytes()
        return json.loads(content)

    def _load_index_parameters_from_json_file(self):
        tenant_object = (
            f"{self.index_mapping_object_prefix.strip('/')}/"
            f"{self.project_name}.json"
        )

        try:
            index_parameters = self._download_json_file(tenant_object)
        except NotFound:
            self.logger.info(
                "Per-tenant index metadata %s was not found; using legacy mapping %s",
                tenant_object,
                self.index_mapping_json_file,
            )
            try:
                legacy_mapping = self._download_json_file(
                    self.index_mapping_json_file
                )
                index_parameters = legacy_mapping.get(self.project_name)
            except (GoogleAPICallError, json.JSONDecodeError) as exc:
                raise DownloadJsonError(
                    f"Failed to read index metadata for {self.project_name}"
                ) from exc
        except (GoogleAPICallError, json.JSONDecodeError) as exc:
            raise DownloadJsonError(
                f"Failed to read index metadata for {self.project_name}"
            ) from exc

        if not isinstance(index_parameters, dict):
            raise DownloadJsonError(
                f"Index metadata for {self.project_name} is missing or invalid"
            )

        missing = self.required_index_parameters.difference(index_parameters)
        if missing:
            raise DownloadJsonError(
                "Index metadata for "
                f"{self.project_name} is missing: {', '.join(sorted(missing))}"
            )

        return index_parameters

    def setup(self):
        # initialize the post init process
        self._post_init_()
        # Initialize embedding model for text -> vector
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embeddings_model_name,
                                                       project=self.project_id,
                                                       vertexai=True)
        self.logger.info("Initializing Indexer setup...")

        # Init Google Vertex AI environment with project + region + bucket
        aiplatform.init(project=self.project_id, location=self.index_region, staging_bucket=self.bucket_uri)
        self.logger.debug(f"Vertex AI initialized (project={self.project_id}, region={self.index_region}, bucket={self.bucket_uri})")

        # Connect to existing Matching Engine index and endpoint
        self.my_index = aiplatform.MatchingEngineIndex(self.index_id)
        self.my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(self.index_endpoint_id)
        self.logger.info(f"Connected to Matching Engine index={self.index_id}, endpoint={self.index_endpoint_id}")

        # Create a LangChain VectorStore that connects to GCP Matching Engine
        self.store = VectorSearchVectorStore.from_components(
            project_id=self.project_id,
            region=self.index_region,
            gcs_bucket_name=self.vector_bucket,
            index_id=self.my_index.name,
            endpoint_id=self.my_index_endpoint.name,
            embedding=self.embeddings,
            stream_update=True,  # insert vectors as they arrive
        )
        self.logger.info("VectorSearchVectorStore created successfully.")
        return self

    async def ingest_data(self):
        if len(self.text) != len(self.meta_data):
            raise ValueError("Every indexed chunk must have metadata")
        chunk_ids = []
        for metadata in self.meta_data:
            if not metadata.get("file_name") or metadata.get("chunk_number") is None:
                raise ValueError("Chunk metadata must identify its source file and chunk number")
            identity = [self.project_name, metadata["file_name"], metadata["chunk_number"]]
            chunk_ids.append(sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest())
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk identities must be unique within a document")
        all_ids = []
        total_batches = (len(self.text) + self.batch_size - 1) // self.batch_size
        self.logger.info(f"Starting ingestion of {len(self.text)} texts in {total_batches} batches...")

        # Insert texts in batches into the vector store
        for batch_idx, i in enumerate(range(0, len(self.text), self.batch_size), start=1):
            # Slice current batch of texts and metadata
            batch_texts = self.text[i : i + self.batch_size]
            batch_meta  = self.meta_data[i : i + self.batch_size]

            self.logger.debug(f"Processing batch {batch_idx}/{total_batches} (size={len(batch_texts)})")

            try:
                # Add batch to the vector DB (async call)
                batch_ids = await self.store.aadd_texts(
                    texts=batch_texts,
                    metadatas=batch_meta,
                    ids=chunk_ids[i : i + self.batch_size],
                )
                all_ids.extend(batch_ids)

                # Log insertion result
                self.logger.info(
                    f"✓ Inserted {len(batch_ids)} vectors "
                    f"({len(all_ids)}/{len(self.text)}) after batch {batch_idx}"
                )
            except Exception as e:
                self.logger.error(f"Failed to insert batch {batch_idx}: {e}")
                raise

        self.logger.info(f"Ingestion complete: total {len(all_ids)} vectors inserted.")
