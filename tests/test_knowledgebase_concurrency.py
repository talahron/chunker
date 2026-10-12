import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from google.api_core.exceptions import PreconditionFailed

from src.agents.caching_agents import DocumentCachingProcessor


class KnowledgebaseConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = DocumentCachingProcessor.__new__(DocumentCachingProcessor)
        self.processor.logger = MagicMock()

    def test_save_uses_observed_generation(self):
        blob = MagicMock()
        blob.generation = 17
        blob.bucket.name = "knowledgebase"
        blob.name = "general.md"

        self.processor._save_knowledgebase(blob, "updated")

        blob.upload_from_string.assert_called_once_with(
            "updated",
            content_type="text/plain",
            if_generation_match=17,
        )

    async def test_conflict_refreshes_context_and_retries(self):
        stale_blob = SimpleNamespace(name="general.md")
        fresh_blob = SimpleNamespace(name="general.md")
        self.processor._save_knowledgebase = MagicMock(
            side_effect=[PreconditionFailed("changed"), None]
        )
        self.processor._prepare_knowledgebase = MagicMock(return_value=fresh_blob)
        self.processor.run_knowledgebase_agent = AsyncMock(
            return_value="fresh update"
        )

        await self.processor._save_knowledgebase_with_retry(
            agent_name="general",
            agent_prompt="prompt",
            gcs_uri="gs://pdf/document.pdf",
            knowledgebase_blob=stale_blob,
            content="stale update",
            cached_content=None,
        )

        self.assertEqual(self.processor._save_knowledgebase.call_count, 2)
        self.processor._prepare_knowledgebase.assert_called_once_with("general.md")
        self.processor.run_knowledgebase_agent.assert_awaited_once()
        self.processor._save_knowledgebase.assert_called_with(
            fresh_blob,
            "fresh update",
        )


if __name__ == "__main__":
    unittest.main()
