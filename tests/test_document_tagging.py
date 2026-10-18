import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


from src.agents.caching_agents import DOCUMENT_TAG_SCHEMAS, DocumentCachingProcessor
from src.chunking_types.chunking_utils import ChunkerUtils
from src.exceptions import ModelValidateSchemaError
from src.pydantic_models import (
    DocumentClassification,
    DocumentClassificationTag,
    EvidenceScale,
    EvidenceScaleTag,
    ParagraphData,
    ParagraphPageNumber,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def make_chunk(object_id):
    return ParagraphData(
        object_id=object_id,
        file_name="document.pdf",
        type=[],
        pages=[ParagraphPageNumber(page_number=1)],
        content=f"chunk {object_id}",
        location=[],
    )


def make_processor():
    processor = DocumentCachingProcessor.__new__(DocumentCachingProcessor)
    processor.logger = MagicMock()
    processor.model_name = "model"
    processor.blob_path_pdf = "document.pdf"
    processor.thinking_budget_enrichment_chunks = 128
    processor.max_concurrent = 10
    processor.document_tagging_agents = [
        ("document_classification", "subject prompt"),
        ("evidence_scale", "type prompt"),
    ]
    return processor


class DocumentTagModelsTest(unittest.TestCase):
    def test_paragraph_defaults_to_empty_tags(self):
        chunk = make_chunk(1.0)
        self.assertIsNone(chunk.document_classification)
        self.assertIsNone(chunk.evidence_scale)

    def test_every_tag_name_matches_its_schema_and_a_paragraph_field(self):
        for tag_name, tag_schema in DOCUMENT_TAG_SCHEMAS.items():
            self.assertEqual(list(tag_schema.model_fields), [tag_name])
            self.assertIn(tag_name, ParagraphData.model_fields)

    def test_tag_rejects_values_outside_the_enum(self):
        with self.assertRaises(ValueError):
            DocumentClassificationTag(document_classification="not a subject")

    def test_prompt_files_exist(self):
        for name in ("document_classification.md", "evidence_scale.md"):
            self.assertTrue((PROMPTS_DIR / name).read_text(encoding="utf-8").strip())


class TagDocumentTest(unittest.IsolatedAsyncioTestCase):
    async def test_merges_the_tags_of_all_agents(self):
        processor = make_processor()
        processor.run_document_tagging_agent = AsyncMock(side_effect=[
            DocumentClassificationTag(document_classification=DocumentClassification.LABOR_LAW),
            EvidenceScaleTag(evidence_scale=EvidenceScale.VERDICT),
        ])

        tags = await processor.tag_document(gcs_uri="gs://pdf/document.pdf")

        self.assertEqual(tags, {
            "document_classification": DocumentClassification.LABOR_LAW,
            "evidence_scale": EvidenceScale.VERDICT,
        })

    async def test_failed_agent_leaves_only_its_tag_empty(self):
        processor = make_processor()
        processor.run_document_tagging_agent = AsyncMock(side_effect=[
            ModelValidateSchemaError("bad response"),
            EvidenceScaleTag(evidence_scale=EvidenceScale.CONTRACT),
        ])

        tags = await processor.tag_document(gcs_uri="gs://pdf/document.pdf")

        self.assertEqual(tags, {"evidence_scale": EvidenceScale.CONTRACT})

    async def test_agent_uses_cache_and_enum_schema(self):
        processor = make_processor()
        generate = AsyncMock(return_value=SimpleNamespace(
            parsed=DocumentClassificationTag(document_classification=DocumentClassification.CIVIL_LAW)
        ))
        processor.gemini_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        )

        result = await processor.run_document_tagging_agent(
            tag_name="document_classification",
            agent_prompt="subject prompt",
            gcs_uri="gs://pdf/document.pdf",
            cached_content=SimpleNamespace(name="caches/1"),
        )

        self.assertEqual(result.document_classification, DocumentClassification.CIVIL_LAW)
        config = generate.call_args.kwargs["config"]
        self.assertEqual(config.cached_content, "caches/1")
        self.assertIs(config.response_schema, DocumentClassificationTag)
        # with a cache the PDF is not attached again
        self.assertEqual(len(generate.call_args.kwargs["contents"][0].parts), 1)

    async def test_agent_attaches_pdf_without_cache_and_retries_invalid_responses(self):
        processor = make_processor()
        generate = AsyncMock(side_effect=[
            SimpleNamespace(parsed=None),
            SimpleNamespace(parsed=EvidenceScaleTag(evidence_scale=EvidenceScale.DECISION)),
        ])
        processor.gemini_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
        )
        processor.run_document_tagging_agent.retry.sleep = AsyncMock()

        result = await processor.run_document_tagging_agent(
            tag_name="evidence_scale",
            agent_prompt="type prompt",
            gcs_uri="gs://pdf/document.pdf",
        )

        self.assertEqual(result.evidence_scale, EvidenceScale.DECISION)
        self.assertEqual(generate.await_count, 2)
        parts = generate.call_args.kwargs["contents"][0].parts
        self.assertEqual(parts[1].file_data.file_uri, "gs://pdf/document.pdf")


class PipelineTaggingTest(unittest.IsolatedAsyncioTestCase):
    async def test_document_tags_reach_every_chunk_before_enrichment(self):
        processor = make_processor()
        processor.knowledgebase_agents = []
        processor._upload_pdf_file_to_memory = MagicMock(return_value=("gs://pdf/document.pdf", None))
        processor._count_tokens = AsyncMock(return_value=100)
        processor._prepare_knowledgebase = MagicMock()
        # the subject agent failed - a value guessed earlier by the chunking LLM must be cleared
        processor.tag_document = AsyncMock(return_value={"evidence_scale": EvidenceScale.VERDICT})
        guessed = make_chunk(2.0)
        guessed.document_classification = DocumentClassification.OTHER

        async def enrich_chunk(*, chunk, **_):
            self.assertEqual(chunk.evidence_scale, EvidenceScale.VERDICT)
            return chunk

        processor.enrich_chunk = enrich_chunk

        chunks = await processor.enrich_chunks_pipeline(chunks=[guessed, make_chunk(1.0)])

        self.assertEqual([c.evidence_scale for c in chunks], [EvidenceScale.VERDICT] * 2)
        self.assertEqual([c.document_classification for c in chunks], [None, None])
        processor.tag_document.assert_awaited_once_with(
            gcs_uri="gs://pdf/document.pdf", cached_content=None
        )


class BuildMetadataTest(unittest.IsolatedAsyncioTestCase):
    async def test_tags_are_flattened_into_chunk_metadata(self):
        utils = ChunkerUtils()
        utils.logger = MagicMock()
        utils.file_config = SimpleNamespace(CUSTOMER="TNUFA", DISTRICT="NORTH", CASE_NUMBER="2")
        tagged = make_chunk(1.0)
        tagged.document_classification = DocumentClassification.CIVIL_LAW
        tagged.evidence_scale = EvidenceScale.CONTRACT
        partial = make_chunk(2.0)
        partial.evidence_scale = EvidenceScale.CONTRACT

        _, metadata = await utils._build_content_and_metadata([tagged, partial])

        self.assertEqual(metadata[0], {
            "file_name": "document.pdf",
            "chunk_number": 1.0,
            "pages": "[1]",
            "document_classification": "משפט אזרחי",
            "evidence_scale": "חוזה",
            "customer": "TNUFA",
            "district": "NORTH",
            "case_number": "2",
        })
        self.assertNotIn("document_classification", metadata[1])
        self.assertIs(type(metadata[0]["evidence_scale"]), str)

    async def test_vector_search_receives_tags_as_restricts(self):
        from langchain_google_vertexai.vectorstores._utils import to_data_points

        utils = ChunkerUtils()
        utils.logger = MagicMock()
        utils.file_config = SimpleNamespace(CUSTOMER="TNUFA", DISTRICT="NORTH", CASE_NUMBER="2")
        chunk = make_chunk(1.0)
        chunk.document_classification = DocumentClassification.LABOR_LAW
        chunk.evidence_scale = EvidenceScale.PROCEDURE
        _, metadata = await utils._build_content_and_metadata([chunk])

        [point] = to_data_points(ids=["id"], embeddings=[[0.1]], metadatas=metadata)

        restricts = {r.namespace: list(r.allow_list) for r in point.restricts}
        self.assertEqual(restricts["document_classification"], ["דיני עבודה"])
        self.assertEqual(restricts["evidence_scale"], ["נוהל"])
        self.assertEqual(restricts["case_number"], ["2"])


if __name__ == "__main__":
    unittest.main()
