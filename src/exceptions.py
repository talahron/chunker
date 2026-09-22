"""
Custom Exception classes for the structured data extractor
"""
import asyncio
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from google.genai import errors as genai_errors

# HTTP statuses where a retry has a real chance of succeeding
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only — never retry errors that will fail identically."""
    # Raw genai client (DocumentCachingProcessor, ContextSentenceSplitAgent)
    if isinstance(exc, genai_errors.APIError):
        return exc.code in _RETRYABLE_STATUS
    # pydantic-ai agents (AgenticTextAgent, SentenceSplitAgent, etc.)
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, UsageLimitExceeded):
        return False
    if isinstance(exc, UnexpectedModelBehavior):
        return True   # bad/empty structured output — nondeterministic, worth another try
    if isinstance(exc, ModelValidateSchemaError):
        return True
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True
    return False


class MaxRetriesAttemptsError(Exception):
    """Error reached the maximum number of retries when uploading or downloading from GCS."""
    pass

class ObjectJsonDownloadError(Exception):
    """Error downloading object json from GCS"""
    pass

class ChunkingDataError(Exception):
    """Error failed the whole process of chunk the input json"""
    pass

class IngestToVectorError(Exception):
    """Error Failed to ingest the Chunk to vector index"""
    pass

class AddObjectError(Exception):
    """Error Failed to insert object into the chunks object"""
    pass

class LargeChunksSplitAgentError(Exception):
    """Error Failed to Split Large Chunks from the Agent"""
    pass

class SummarizerAgentError(Exception):
    """Error Failed to Summarize Chunks"""
    pass

class AgenticChunkingError(Exception):
    """Error Failed to Summarize Chunks"""
    pass

class BatchPredictionError(Exception):
    """Error to create batch of text to process"""
    pass

class PipelineTaskExecutorError(Exception):
    """Pipeline execution failure"""
    pass

class DownloadJsonError(Exception):
    pass

class DownloadPdfFromGCPBuckerError(Exception):
    pass

class ContextCachingError(Exception):
    pass


class GeminiCachingEnrichmentError(Exception):
    pass


class ModelValidateSchemaError(Exception):
    pass
