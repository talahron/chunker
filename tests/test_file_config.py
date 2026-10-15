import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError

from src.agents.caching_agents import DocumentCachingProcessor
from src.file_config import FileConfig, load_file_config
from src.Inserts_data_into_vector_db import Indexer
from src.pydantic_models import ChunkingStrategy

# Fixed values, so editing dummy_file_config.json for manual checks never breaks the tests
BASE_CONFIG = {
    "INDEX_END_POINT_ID": "5790520813077659648",
    "INDEX_ID": "7766765420656197632",
    "PROJECT_ID": "moj-gcp-d-rag",
    "PROJECT_NUMBER": "458040623421",
    "CUSTOMER": "TNUFA",
    "DISTRICT": "NORTH",
    "CASE_NUMBER": "1",
    "EXTRACTION_MODEL": "QWEN",
    "CHUNKING_TYPE": "SEMANTIC",
    "DOCUMENT_CLASSIFICATION": True,
    "EVIDENCE_SCALE": True,
    "EXTENDED_KNOWLEDGEBASE": True,
}


def make_config(**overrides):
    return FileConfig.model_validate({**BASE_CONFIG, **overrides})


class FileConfigTest(unittest.TestCase):
    def test_dummy_config_loads(self):
        self.assertIsInstance(asyncio.run(load_file_config("folder/document.json")), FileConfig)

    def test_empty_file_name_is_rejected(self):
        with self.assertRaises(ValueError):
            asyncio.run(load_file_config(""))

    def test_ids_stay_exact(self):
        config = make_config(INDEX_ID=7766765420656197632)
        self.assertEqual(config.INDEX_ID, "7766765420656197632")
        for bad_id in (7.766765420656198e+18, "7.766765420656198e+18", "abc"):
            with self.subTest(bad_id=bad_id), self.assertRaises(ValidationError):
                make_config(INDEX_ID=bad_id)

    def test_every_chunking_type_maps_to_its_strategy(self):
        for strategy in ChunkingStrategy:
            self.assertIs(make_config(CHUNKING_TYPE=strategy.value.upper()).CHUNKING_TYPE, strategy)
        with self.assertRaises(ValidationError):
            make_config(CHUNKING_TYPE="NOT_A_TYPE")

    def test_switches_choose_the_agents(self):
        config = make_config(EXTENDED_KNOWLEDGEBASE=False, DOCUMENT_CLASSIFICATION=False)
        self.assertTrue(config.runs_agent("general"))
        self.assertFalse(config.runs_agent("extended"))
        self.assertFalse(config.runs_agent("document_classification"))
        self.assertTrue(config.runs_agent("evidence_scale"))
        with self.assertRaises(KeyError):
            config.runs_agent("extened")

    def test_knowledgebase_file_name_per_case(self):
        self.assertEqual(make_config().knowledgebase_file_name("general"), "general_1.md")
        self.assertEqual(make_config(CUSTOMER="OTHER", CASE_NUMBER="").knowledgebase_file_name("extended"), "extended.md")

    def test_index_parameters(self):
        self.assertEqual(make_config().index_parameters, {
            "INDEX_ID": "7766765420656197632",
            "INDEX_END_POINT_ID": "5790520813077659648",
            "PROJECT_NUMBER": "458040623421",
        })


class DocumentAgentsRouteTest(unittest.IsolatedAsyncioTestCase):
    @patch("src.agents.caching_agents.storage.Client")
    @patch.object(DocumentCachingProcessor, "_initialize_client")
    def test_disabled_agents_are_not_loaded(self, _client, _storage):
        processor = DocumentCachingProcessor(
            model_name="model",
            project_id="project",
            google_provider_region="region",
            caching_location="location",
            knowledgebase_bucket_name="kb-bucket",
            path_to_enrichment_prompt="prompts/document_context_enrichment.md",
            knowledgebase_agent_prompt_paths=[
                ("general", "prompts/knowledgebase_general.md"),
                ("extended", "prompts/knowledgebase_extended.md"),
            ],
            document_tagging_agent_prompt_paths=[
                ("document_classification", "prompts/document_classification.md"),
                ("evidence_scale", "prompts/evidence_scale.md"),
            ],
            file_config=make_config(EXTENDED_KNOWLEDGEBASE=False, EVIDENCE_SCALE=False),
        )
        self.assertEqual([name for name, _ in processor.knowledgebase_agents], ["general"])
        self.assertEqual([name for name, _ in processor.document_tagging_agents], ["document_classification"])

    async def test_step4_reads_and_writes_the_case_files(self):
        processor = DocumentCachingProcessor.__new__(DocumentCachingProcessor)
        processor.logger = MagicMock()
        processor.file_config = make_config()
        processor.knowledgebase_agents = [("general", "prompt")]
        processor.document_tagging_agents = []
        processor.max_concurrent = 1
        processor.blob_path_pdf = "document.pdf"
        processor._upload_pdf_file_to_memory = MagicMock(return_value=("gs://pdf/document.pdf", None))
        processor._count_tokens = AsyncMock(return_value=100)
        case_blob = MagicMock()
        processor._prepare_knowledgebase = MagicMock(return_value=case_blob)
        processor.run_knowledgebase_agent = AsyncMock(return_value="updated")
        processor._save_knowledgebase_with_retry = AsyncMock()

        await processor.enrich_chunks_pipeline(chunks=[])

        processor._prepare_knowledgebase.assert_called_once_with("general_1.md")
        self.assertIs(processor.run_knowledgebase_agent.call_args.kwargs["knowledgebase_blob"], case_blob)
        self.assertIs(processor._save_knowledgebase_with_retry.call_args.kwargs["knowledgebase_blob"], case_blob)


class IndexerRouteTest(unittest.TestCase):
    @patch("src.Inserts_data_into_vector_db.storage.Client")
    def test_file_config_ids_skip_the_index_mapping(self, storage_client):
        indexer = Indexer(
            text=[], meta_data=[], batch_size=10,
            index_mapping_bucket="rag-index-mapping",
            index_mapping_json_file="index_mapping.json",
            index_mapping_object_prefix="indexes",
            embeddings_model_name="embedding-model",
            project_name="tenant-a",
            vector_bucket="rag-index-tenant-a",
            index_region="me-west1",
            index_parameters=make_config().index_parameters,
        )
        indexer._post_init_()
        self.assertEqual(indexer.index_id, "7766765420656197632")
        self.assertEqual(indexer.index_endpoint_id, "5790520813077659648")
        self.assertEqual(indexer.project_id, "458040623421")
        storage_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
