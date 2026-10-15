import os
import unittest

from pydantic import ValidationError

os.environ.update({
    "GCP_PROJECT_ID": "rag-project",
    "GCP_LOCATION": "eu",
    "CACHING_LOCATION": "europe-west4",
    "EMBEDDINGS_REGION": "europe-west4",
    "INPUT_BUCKET_NAME_INIT": "rag-output-extractor-",
    "INPUT_BUCKET_NAME_INIT_PDF": "rag-output-flipper-",
    "MODEL_NAME": "model",
    "MODEL_NAME_ENRICHMENT_CHUNKER": "model",
    "THINKING_BUDGET_ENRICHMENT_CHUNKER": "128",
    "MAX_TOKENS": "500",
    "MIN_TOKENS": "15",
    "INDEX_MAPPING_BUCKET": "rag-index-mapping",
    "INDEX_MAPPING_JSON_FILE": "index_mapping.json",
    "EMBEDDINGS_MODEL_NAME": "embedding-model",
    "INDEX_REGION": "me-west1",
    "VECTOR_BUCKET_NAME_INIT": "rag-index-",
    "KNOWLEDGEBASE_BUCKET_NAME_INIT": "rag-knowledgebase-",
    "INCREMENT": "10000",
    "BATCH_SIZE": "500",
    "MODEL_NAME_SENTENCE_SPLITTER": "model",
    "MAX_WORKERS_FOR_BATCH_PROCESSING": "5",
    "SEMANTIC_SIMILARITY_THRESHOLD": "0.3",
    "SEMANTIC_MAX_TOKENS_PER_CHUNK": "500",
    "SEMANTIC_MIN_TOKENS_PER_CHUNK": "3",
    "CLUSTER_MIN_TOKENS_PER_SENTENCE": "50",
    "CHARACTER_MAX_CHARS": "2000",
    "CHARACTER_OVERLAP": "200",
})

from src.settings import Settings, settings  # noqa: E402


class WorkerSettingsTest(unittest.TestCase):
    def test_worker_mode_requires_subscription(self):
        values = settings.model_dump()
        values.update(WORKER_MODE=True, WORKER_SUBSCRIPTION="")

        with self.assertRaises(ValidationError):
            Settings(**values)

    def test_worker_lease_default_covers_long_running_jobs(self):
        values = settings.model_dump()
        values.pop("WORKER_MAX_LEASE_SECONDS", None)

        self.assertEqual(Settings(**values).WORKER_MAX_LEASE_SECONDS, 43200)


if __name__ == "__main__":
    unittest.main()
