# MOJ GCP RAG Chunker

Reads the extractor JSON for a document from GCS, splits it into retrieval
chunks, enriches each chunk with context from the original PDF, updates the
dataset knowledge base, and ingests the result into Vertex AI Vector Search.

## Chunking strategies

`CHUNKING_STRATEGY` selects one of:

| value       | class                   | how it splits                                          |
|-------------|-------------------------|--------------------------------------------------------|
| `header`    | `HeaderChunker`         | new chunk at every `section_header`, oversized ones split by an LLM |
| `agentic`   | `AgenticChunker`        | an LLM aggregates parser objects into paragraphs        |
| `semantic`  | `SemanticChunker`       | cumulative: merge sentences while cosine similarity holds |
| `cluster`   | `ClusterSemanticChunker`| global DP over the sentence similarity matrix           |
| `character` | `CharacterChunker`      | fixed-size character windows                            |

## Configuration

All settings come from environment variables, validated by `Settings` in
`src/settings.py`. In deployment they are injected by the Helm chart. For local
runs set `DEV=1` and put the same keys in a `.dev_env` file at the repo root.

Missing a required variable fails at import time, before any work starts.

### Required

`GCP_PROJECT_ID`, `GCP_LOCATION`, `CACHING_LOCATION`, `EMBEDDINGS_REGION`,
`INDEX_REGION`, `PROJECT_NAME`, `FILE_NAME`, `INPUT_BUCKET_NAME_INIT`,
`INPUT_BUCKET_NAME_INIT_PDF`, `VECTOR_BUCKET_NAME_INIT`,
`KNOWLEDGEBASE_BUCKET_NAME_INIT`, `INDEX_MAPPING_BUCKET`,
`INDEX_MAPPING_JSON_FILE`, `MODEL_NAME`, `MODEL_NAME_ENRICHMENT_CHUNKER`,
`MODEL_NAME_SENTENCE_SPLITTER`, `EMBEDDINGS_MODEL_NAME`,
`THINKING_BUDGET_ENRICHMENT_CHUNKER`, `MAX_TOKENS`, `MIN_TOKENS`, `INCREMENT`,
`BATCH_SIZE`, `MAX_WORKERS_FOR_BATCH_PROCESSING`, `CHUNKING_STRATEGY`,
`SEMANTIC_SIMILARITY_THRESHOLD`, `SEMANTIC_MAX_TOKENS_PER_CHUNK`,
`SEMANTIC_MIN_TOKENS_PER_CHUNK`, `CLUSTER_MIN_TOKENS_PER_SENTENCE`,
`CHARACTER_MAX_CHARS`.

`CACHING_LOCATION` is also accepted under the name `CACHING_REGION`.

### Derived, do not set

These are computed in `Settings.__init__` and any supplied value is discarded:

- `INPUT_BUCKET_NAME` = `INPUT_BUCKET_NAME_INIT` + `PROJECT_NAME`
- `INPUT_BUCKET_NAME_PDF` = `INPUT_BUCKET_NAME_INIT_PDF` + `PROJECT_NAME`
- `VECTOR_BUCKET_NAME` = `VECTOR_BUCKET_NAME_INIT` + `PROJECT_NAME`
- `KNOWLEDGEBASE_BUCKET_NAME` = `KNOWLEDGEBASE_BUCKET_NAME_INIT` + `PROJECT_NAME`
- `FILE_NAME_PDF` = stem of `FILE_NAME` + `.pdf`

### Per-execution

`PROJECT_NAME` and `FILE_NAME` identify the document to process, so they must be
supplied per job execution. Every bucket path above is built from them, and an
empty value produces bucket and object names that do not exist.

### Optional

- `GOOGLE_APPLICATION_CREDENTIALS` - read only when `DEV` is set. In GCP the
  attached service account is used through Application Default Credentials.
- `CHARACTER_OVERLAP` - reserved, not read by any chunker today.

## Logging

Logs go to stdout, which is what Cloud Logging collects. A rotating file handler
is added as well when the log directory is writable, and is skipped with a
warning when it is not.

## Running

```bash
# local
DEV=1 python3 main.py

# container
docker build -t chunker . && docker run --env-file .dev_env chunker
```
