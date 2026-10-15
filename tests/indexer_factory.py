from src.Inserts_data_into_vector_db import Indexer


def make_indexer():
    return Indexer(
        text=[],
        meta_data=[],
        batch_size=10,
        index_parameters={
            "INDEX_ID": "1",
            "PROJECT_NUMBER": "2",
            "INDEX_END_POINT_ID": "3",
        },
        embeddings_model_name="embedding-model",
        project_name="tenant-a",
        vector_bucket="rag-index-tenant-a",
        index_region="me-west1",
    )
