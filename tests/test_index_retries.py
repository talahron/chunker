import unittest
from unittest.mock import AsyncMock

from test_index_metadata import make_indexer


class IndexRetryTest(unittest.IsolatedAsyncioTestCase):
    def make_indexer(self, tenant="tenant-a"):
        indexer = make_indexer()
        indexer.project_name = tenant
        indexer.batch_size = 1
        indexer.text = ["first chunk", "second chunk"]
        indexer.meta_data = [
            {"file_name": "folder/document.pdf", "chunk_number": number}
            for number in (0, 1)
        ]
        indexer.store = AsyncMock()
        indexer.store.aadd_texts.return_value = ["inserted"]
        return indexer

    async def test_retries_reuse_vector_ids_across_batches(self):
        indexer = self.make_indexer()
        await indexer.ingest_data()
        first_ids = [call.kwargs["ids"] for call in indexer.store.aadd_texts.call_args_list]
        indexer.store.reset_mock()
        await indexer.ingest_data()
        second_ids = [call.kwargs["ids"] for call in indexer.store.aadd_texts.call_args_list]
        self.assertEqual(first_ids, second_ids)
        self.assertNotEqual(first_ids[0], first_ids[1])

    async def test_tenants_and_source_files_have_distinct_ids(self):
        first = self.make_indexer()
        second = self.make_indexer("tenant-b")
        third = self.make_indexer()
        for metadata in third.meta_data:
            metadata["file_name"] = "other/document.pdf"
        for indexer in (first, second, third):
            await indexer.ingest_data()
        ids = [indexer.store.aadd_texts.call_args.kwargs["ids"][0] for indexer in (first, second, third)]
        self.assertEqual(len(set(ids)), 3)

    async def test_missing_or_duplicate_identity_fails_before_writing(self):
        for invalid in ({}, {"file_name": "document.pdf"}):
            indexer = self.make_indexer()
            indexer.meta_data[0] = invalid
            with self.assertRaises(ValueError):
                await indexer.ingest_data()
            indexer.store.aadd_texts.assert_not_called()
        indexer = self.make_indexer()
        indexer.meta_data[1] = indexer.meta_data[0]
        with self.assertRaises(ValueError):
            await indexer.ingest_data()
        indexer.store.aadd_texts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
