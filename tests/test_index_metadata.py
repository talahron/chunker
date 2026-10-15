import json
import unittest
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound

from src.Inserts_data_into_vector_db import Indexer
from src.exceptions import DownloadJsonError


def make_indexer():
    return Indexer(
        text=[],
        meta_data=[],
        batch_size=10,
        index_mapping_bucket="rag-index-mapping",
        index_mapping_json_file="index_mapping.json",
        index_mapping_object_prefix="indexes",
        embeddings_model_name="embedding-model",
        project_name="tenant-a",
        vector_bucket="rag-index-tenant-a",
        index_region="me-west1",
    )


class IndexMetadataTest(unittest.TestCase):
    @patch("src.Inserts_data_into_vector_db.storage.Client")
    def test_prefers_per_tenant_metadata(self, client_class):
        payload = {
            "INDEX_ID": 1,
            "PROJECT_NUMBER": 2,
            "INDEX_END_POINT_ID": 3,
        }
        bucket = client_class.return_value.bucket.return_value
        bucket.blob.return_value.download_as_bytes.return_value = json.dumps(
            payload
        ).encode()

        result = make_indexer()._load_index_parameters_from_json_file()

        self.assertEqual(result, payload)
        bucket.blob.assert_called_once_with("indexes/tenant-a.json")

    @patch("src.Inserts_data_into_vector_db.storage.Client")
    def test_falls_back_to_legacy_mapping(self, client_class):
        payload = {
            "INDEX_ID": 1,
            "PROJECT_NUMBER": 2,
            "INDEX_END_POINT_ID": 3,
        }
        tenant_blob = MagicMock()
        tenant_blob.download_as_bytes.side_effect = NotFound("missing")
        legacy_blob = MagicMock()
        legacy_blob.download_as_bytes.return_value = json.dumps(
            {"tenant-a": payload}
        ).encode()
        bucket = client_class.return_value.bucket.return_value
        bucket.blob.side_effect = [tenant_blob, legacy_blob]

        result = make_indexer()._load_index_parameters_from_json_file()

        self.assertEqual(result, payload)
        self.assertEqual(
            [call.args[0] for call in bucket.blob.call_args_list],
            ["indexes/tenant-a.json", "index_mapping.json"],
        )

    @patch("src.Inserts_data_into_vector_db.storage.Client")
    def test_rejects_incomplete_metadata(self, client_class):
        bucket = client_class.return_value.bucket.return_value
        bucket.blob.return_value.download_as_bytes.return_value = b'{"INDEX_ID": 1}'

        with self.assertRaises(DownloadJsonError):
            make_indexer()._load_index_parameters_from_json_file()


if __name__ == "__main__":
    unittest.main()
