from langchain_google_vertexai import VectorSearchVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.exceptions import DownloadJsonError
from google.cloud import aiplatform
from google.cloud import storage
import json
from src.logger import setup_logger

class Indexer:
    def __init__(self,
                 text,
                 meta_data,
                 batch_size,
                 index_mapping_bucket,
                 index_mapping_json_file,
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
        index_parameters = self._load_index_parameters_from_json_file()
        # extract the index parameters from the json file for the project name
        try:
            self.logger.info(f"Downloading index parameters for project - {self.project_name}")
            index_params_for_project = index_parameters.get(self.project_name)
            self.index_id = index_params_for_project.get("INDEX_ID")
            self.project_id = index_params_for_project.get("PROJECT_NUMBER")
            self.index_endpoint_id = index_params_for_project.get("INDEX_END_POINT_ID")
            self.logger.info(f"Successfully downloaded index parameters for project - {self.project_name}")
            # make sure that the parameters are of type string - aiplatform expect type string
            self.index_id = self.index_id if isinstance(self.index_id, str) else str(self.index_id)
            self.project_id = self.project_id if isinstance(self.project_id, str) else str(self.project_id)
            self.index_endpoint_id = self.index_endpoint_id if isinstance(self.index_endpoint_id, str) else str(self.index_endpoint_id)
        except DownloadJsonError as e:
            self.logger.error(f"Failed to download index parameters for project - {self.project_name}")
            raise

    def _load_index_parameters_from_json_file(self):
        client = storage.Client()
        bucket = client.get_bucket(self.index_mapping_bucket)
        blob = bucket.blob(self.index_mapping_json_file)
        # Download the file content as a string
        content = blob.download_as_string()
        # Parse JSON content
        return json.loads(content)

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
                batch_ids = await self.store.aadd_texts(texts=batch_texts, metadatas=batch_meta)
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
