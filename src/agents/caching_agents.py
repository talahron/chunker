from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from google import genai
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.exceptions import (
    DownloadPdfFromGCPBuckerError,
    ModelValidateSchemaError,
    _is_retryable,
)
from src.logger import setup_logger
from src.pydantic_models import (
    AgenticResponse,
    DocumentClassificationTag,
    EvidenceScaleTag,
    FileConfig,
    ParagraphData,
    ParagraphLocation,
    ParagraphPageNumber,
    ParagraphType,
    SimpleAgenticResponse,
)

logger = setup_logger(__name__)
KNOWLEDGEBASE_UPDATE_ATTEMPTS = 3

# Response schema of each document tagging agent, keyed by the ParagraphData field it fills
DOCUMENT_TAG_SCHEMAS = {
    "document_classification": DocumentClassificationTag,
    "evidence_scale": EvidenceScaleTag,
}

# Chunk enrichment reads legal documents, so the safety filters stay off
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

# ----- Agent 4: Context Caching with Gemini for Chunk Enrichment
@dataclass
class DocumentCachingProcessor:
    """Run document-level tasks and chunk enrichment using an optional shared cache."""
    model_name: str
    project_id: str
    google_provider_region: str
    caching_location: str
    knowledgebase_bucket_name: str
    path_to_enrichment_prompt: Path
    knowledgebase_agent_prompt_paths: list[tuple[str, Path]]
    document_tagging_agent_prompt_paths: list[tuple[str, Path]]
    file_config: FileConfig
    thinking_budget_enrichment_chunks: int = 128
    max_concurrent: int = 10
    bucket_name_pdf: Optional[str] = None
    blob_path_pdf: Optional[str] = None

    def __post_init__(self):
        self.enrichment_prompt = Path(self.path_to_enrichment_prompt).read_text(
            encoding="utf-8"
        )
        self.knowledgebase_agents = [
            (agent_name, Path(prompt_path).read_text(encoding="utf-8"))
            for agent_name, prompt_path in self.knowledgebase_agent_prompt_paths
            if self.file_config.runs_agent(agent_name)
        ]
        self.document_tagging_agents = [
            (tag_name, Path(prompt_path).read_text(encoding="utf-8"))
            for tag_name, prompt_path in self.document_tagging_agent_prompt_paths
            if self.file_config.runs_agent(tag_name)
        ]
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info(
            "Document agents for this file: knowledge base=%s, tagging=%s",
            [agent_name for agent_name, _ in self.knowledgebase_agents],
            [tag_name for tag_name, _ in self.document_tagging_agents],
        )
        self.gemini_client = self._initialize_client()
        self.storage_client = storage.Client()

    def _initialize_client(self):
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.caching_location,
            http_options=HttpOptions(
                base_url=f"https://aiplatform.{self.caching_location}.rep.googleapis.com",
            ),
        )
        api = client._api_client
        logger.info("httpx clients: sync_hooks=%s async_hooks=%s",
                    getattr(api, "_httpx_client", None) and api._httpx_client.event_hooks,
                    getattr(api, "_async_httpx_client", None) and api._async_httpx_client.event_hooks)
        logger.info("GoogleProvider ready (hooks registered on sync + async clients)")
        return client



    def _upload_pdf_file_to_memory(self):
        """Upload PDF file to memory"""
        try:
            self.logger.info(f"Uploading {self.blob_path_pdf} to memory")
            bucket = self.storage_client.bucket(self.bucket_name_pdf)
            blob = bucket.blob(self.blob_path_pdf)
            gcs_uri = f"gs://{bucket.name}/{blob.name}"
            self.logger.info(f"PDF uploaded to GCS: {gcs_uri}")
            return gcs_uri, blob
        except ValueError as e:
            self.logger.error(f"✗ Error uploading {self.blob_path_pdf}: {e}")
            raise

    async def _count_tokens(self, gcs_uri: Optional[storage.Blob] | str = None) -> int:
        """count the number of tokens in the pdf to allow context caching - minimum count should be 1024"""
        self.logger.info("counting tokens for pdf file")
        response = await self.gemini_client.aio.models.count_tokens(
            model=self.model_name,
            contents=types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=gcs_uri,
                        mime_type="application/pdf"
                    ),
                ],
            ),
        )
        self.logger.info(f"total token number is {response.total_tokens}")
        above_threshold = response.total_tokens > 1024
        self.logger.info(
            "total tokens - %s is %s%s",
            response.total_tokens,
            "higher then 1024" if response.total_tokens >= 1024 else "less then 1024",
            " - proceeding with caching" if above_threshold else " cannot proceed with caching",
        )
        return response.total_tokens

    def calculate_ttl_minutes(self,
                              num_chunks: int,
                              avg_seconds_per_chunk: float = 10.0,
                              buffer_multiplier: float = 1.5,
                              min_ttl_minutes: int = 5,
                              max_ttl_minutes: int = 120
                              ) -> int:
        """
        Estimate cache TTL based on chunk count and concurrency.

        The effective throughput is: max_concurrent / avg_seconds_per_chunk
        Total time ≈ (num_chunks / max_concurrent) * avg_seconds_per_chunk
        Then we add a safety buffer for retries, rate limits, cold starts, etc.
        """
        self.logger.info("calculating ttl minutes for pdf file")
        estimated_seconds = (num_chunks / self.max_concurrent) * avg_seconds_per_chunk
        buffered_seconds = estimated_seconds * buffer_multiplier
        ttl_minutes = max(min_ttl_minutes, int(buffered_seconds / 60) + 1)
        ttl_minutes = min(ttl_minutes, max_ttl_minutes)
        self.logger.info(f"Successfully created ttl minutes for pdf file - required ttl time: {ttl_minutes} minutes")

        return ttl_minutes

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),  # Add maximum attempts
           retry=retry_if_exception(_is_retryable),
           reraise=True)  # Re-raise exceptions
    async def create_pdf_cache(self,
                               ttl_minutes: Optional[int] = None,
                               gcs_uri: Optional[storage.Blob] | str = None) -> Any:
        # Create cached content with the PDF baked in if ttl is not none
        if ttl_minutes is not None:
            try:
                self.logger.info(f"Creating cache with file_uri: {self.blob_path_pdf}")
                self.logger.info(f" ttl_minutes is not none - Creating cache for {self.blob_path_pdf}")
                cached_content = await self.gemini_client.aio.caches.create(
                    model=self.model_name,
                    config=types.CreateCachedContentConfig(
                        display_name=f"pdf-context-cache-{Path(self.blob_path_pdf).stem}",
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    # Using the URI of the uploaded file
                                    types.Part.from_uri(
                                        file_uri=gcs_uri,
                                        mime_type="application/pdf"
                                    ),
                                ]
                            )
                        ],
                        # TTL keeps the cache alive for the duration of your processing
                        ttl=f"{ttl_minutes * 60}s"
                        ),
                )
                self.logger.info(f"Cache created: {cached_content.name} for {self.blob_path_pdf}")
                self.logger.info(f"Cache for {self.blob_path_pdf} Expires at:    {cached_content.expire_time}")
                return cached_content
            except genai_errors.APIError as e:
                self.logger.error(f"✗ Error creating cache ({e.code}): {e.message}")
                raise
        else:
            return None

    async def _generate(self,
                        *,
                        parts: list[types.Part],
                        gcs_uri: str,
                        cached_content: Optional[types.CachedContent],
                        **config_options) -> types.GenerateContentResponse:
        """Call Gemini over the PDF: read it from the cache when one exists, otherwise attach it.
        The PDF comes before the request parts in both cases, as the cache puts it."""
        if cached_content is None:
            parts = [types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf"), *parts]
        return await self.gemini_client.aio.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                cached_content=cached_content.name if cached_content is not None else None,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget_enrichment_chunks
                ),
                **config_options,
            ),
        )

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),  # Add maximum attempts
           retry=retry_if_exception(_is_retryable),
           reraise=True)  # Re-raise exceptions
    async def enrich_chunk(self,
                           *,
                           semaphore: asyncio.Semaphore,
                           gcs_uri: str,
                           cached_content: Optional[types.CachedContent] = None,
                           chunk: ParagraphData) -> ParagraphData:
        """Call Gemini with the PDF + one chunk to get contextual grounding."""
        chunk_label = f"{chunk.file_name} / {chunk.pages[0].page_number} / {chunk.object_id}"
        try:
            self.logger.info(f"Running enrichment process for chunk - {chunk_label}")
            async with semaphore:
                response = await self._generate(
                    parts=[types.Part.from_text(text=self.enrichment_prompt.format(chunk=chunk))],
                    gcs_uri=gcs_uri,
                    cached_content=cached_content,
                    safety_settings=SAFETY_SETTINGS,
                    temperature=0.7,
                )
            self.logger.info(f"Successfully enriched chunk - {chunk_label}")
            # update the chunk content with the new-enriched text
            chunk.content = "\n\n\n".join([response.text.strip(), chunk.content])
            return chunk
        except genai_errors.APIError as e:
            if _is_retryable(e):
                raise  # let tenacity handle it
            self.logger.error(f"✗ Non-retryable error enriching chunk {chunk_label} ({e.code}): {e.message}")
            return chunk  # fall back to the un-enriched chunk

    def _prepare_knowledgebase(self, file_name: str) -> storage.Blob:
        """Get file metadata; create an empty file if missing."""
        bucket = self.storage_client.bucket(self.knowledgebase_bucket_name)
        blob = bucket.get_blob(file_name)

        if blob is None:
            blob = bucket.blob(file_name)
            try:
                blob.upload_from_string(
                    "",
                    content_type="text/plain",
                    if_generation_match=0,
                )
            except PreconditionFailed:
                blob = bucket.get_blob(file_name)
                if blob is None:
                    raise

        return blob

    def _save_knowledgebase(self, blob: storage.Blob, content: str) -> None:
        """Replace the knowledge base file with the validated Markdown."""
        if blob.generation is None:
            blob.reload()
        blob.upload_from_string(
            content,
            content_type="text/plain",
            if_generation_match=blob.generation,
        )
        self.logger.info(
            "Knowledge base saved: gs://%s/%s | characters=%d",
            blob.bucket.name,
            blob.name,
            len(content),
        )

    async def _save_knowledgebase_with_retry(
            self,
            *,
            agent_name: str,
            agent_prompt: str,
            gcs_uri: str,
            knowledgebase_blob: storage.Blob,
            content: str,
            cached_content: Optional[types.CachedContent] = None,
    ) -> None:
        """Re-run an agent against fresh context when another worker wins a write."""
        blob = knowledgebase_blob
        updated_content = content

        for attempt in range(1, KNOWLEDGEBASE_UPDATE_ATTEMPTS + 1):
            try:
                self._save_knowledgebase(blob, updated_content)
                return
            except PreconditionFailed:
                if attempt == KNOWLEDGEBASE_UPDATE_ATTEMPTS:
                    raise

                self.logger.warning(
                    "Knowledge base changed while processing %s; refreshing attempt %d/%d",
                    agent_name,
                    attempt + 1,
                    KNOWLEDGEBASE_UPDATE_ATTEMPTS,
                )
                await asyncio.sleep(attempt)
                blob = self._prepare_knowledgebase(blob.name)
                updated_content = await self.run_knowledgebase_agent(
                    agent_name=agent_name,
                    agent_prompt=agent_prompt,
                    gcs_uri=gcs_uri,
                    knowledgebase_blob=blob,
                    cached_content=cached_content,
                )

    async def run_knowledgebase_agent(
            self,
            *,
            agent_name: str,
            agent_prompt: str,
            gcs_uri: str,
            knowledgebase_blob: storage.Blob,
            cached_content: Optional[types.CachedContent] = None,
    ) -> str:
        """Run one knowledge base task; propagate failures to the task group."""
        try:
            parts = [
                types.Part.from_text(text=agent_prompt),
                types.Part.from_text(
                    text=f"source_metadata: {self.blob_path_pdf}\n"
                         "current_dataset_context_md:"
                ),
            ]

            # Attach existing context only when the file has content.
            if knowledgebase_blob.size:
                parts.append(
                    types.Part.from_uri(
                        file_uri=(
                            f"gs://{knowledgebase_blob.bucket.name}/"
                            f"{knowledgebase_blob.name}"
                        ),
                        mime_type="text/plain",
                    )
                )
            else:
                parts.append(types.Part.from_text(text='""'))

            self.logger.info("Starting document agent: %s", agent_name)

            response = await self._generate(
                parts=parts,
                gcs_uri=gcs_uri,
                cached_content=cached_content,
                temperature=0.7,
            )

            content = response.text

            if not content or not content.strip():
                raise ValueError("Knowledge base response is empty.")

            return content

        except Exception as exc:
            exc.add_note(f"Document agent: {agent_name}")
            raise

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    async def run_document_tagging_agent(
            self,
            *,
            tag_name: str,
            agent_prompt: str,
            gcs_uri: str,
            cached_content: Optional[types.CachedContent] = None,
    ) -> Any:
        """Run one tagging task over the whole document; the schema restricts the tag to its enum."""
        self.logger.info("Starting document tagging agent: %s", tag_name)

        response = await self._generate(
            parts=[types.Part.from_text(text=agent_prompt)],
            gcs_uri=gcs_uri,
            cached_content=cached_content,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=DOCUMENT_TAG_SCHEMAS[tag_name],
        )

        # parsed is None when the response does not match the schema (e.g. a value outside the enum)
        if response.parsed is None:
            raise ModelValidateSchemaError(
                f"Document tagging agent {tag_name} returned an invalid response"
            )

        self.logger.info("Document tagging agent %s returned %s", tag_name, response.parsed)
        return response.parsed

    async def tag_document(
            self,
            *,
            gcs_uri: str,
            cached_content: Optional[types.CachedContent] = None,
    ) -> dict[str, Any]:
        """Run all tagging agents concurrently and merge their tags; a failed agent's tag is missing."""
        results = await asyncio.gather(
            *[
                self.run_document_tagging_agent(
                    tag_name=tag_name,
                    agent_prompt=agent_prompt,
                    gcs_uri=gcs_uri,
                    cached_content=cached_content,
                )
                for tag_name, agent_prompt in self.document_tagging_agents
            ],
            return_exceptions=True,
        )

        tags = {}
        for (tag_name, _), result in zip(self.document_tagging_agents, results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"Document tagging agent {tag_name} failed for {self.blob_path_pdf}; "
                    f"continuing without its tag: {result}"
                )
                continue
            tags.update(result.model_dump())

        self.logger.info(f"Document tags for {self.blob_path_pdf}: {tags}")
        return tags

    async def update_knowledgebases(self,
                                    *,
                                    gcs_uri: str,
                                    cached_content: Optional[types.CachedContent] = None) -> None:
        """Run the knowledge base agents, then save their files; a failure is logged and skipped."""
        document_tasks = {}

        try:
            # One file per agent: general.md, or general_<CASE_NUMBER>.md for a per-case customer
            knowledgebase_files = {
                agent_name: self._prepare_knowledgebase(
                    self.file_config.knowledgebase_file_name(agent_name)
                )
                for agent_name, _ in self.knowledgebase_agents
            }

            async with asyncio.TaskGroup() as task_group:
                for agent_name, agent_prompt in self.knowledgebase_agents:
                    document_tasks[agent_name] = task_group.create_task(
                        self.run_knowledgebase_agent(
                            agent_name=agent_name,
                            agent_prompt=agent_prompt,
                            gcs_uri=gcs_uri,
                            knowledgebase_blob=knowledgebase_files[agent_name],
                            cached_content=cached_content,
                        ),
                        name=agent_name,
                    )

            # Save only after all document agents pass validation. Generation
            # preconditions prevent concurrent workers from losing an update.
            async with asyncio.TaskGroup() as task_group:
                for agent_name, agent_prompt in self.knowledgebase_agents:
                    task_group.create_task(
                        self._save_knowledgebase_with_retry(
                            agent_name=agent_name,
                            agent_prompt=agent_prompt,
                            gcs_uri=gcs_uri,
                            knowledgebase_blob=knowledgebase_files[agent_name],
                            content=document_tasks[agent_name].result(),
                            cached_content=cached_content,
                        ),
                        name=f"save-{agent_name}",
                    )

        except* Exception:
            self.logger.exception(
                "Document agents failed; continuing to chunk enrichment."
            )

    async def enrich_chunks(self,
                            *,
                            chunks: List[ParagraphData],
                            gcs_uri: str,
                            cached_content: Optional[types.CachedContent] = None) -> List[ParagraphData]:
        """Enrich all chunks concurrently; a failed chunk is kept un-enriched. Returns them sorted by object id."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = await asyncio.gather(
            *[
                self.enrich_chunk(
                    semaphore=semaphore,
                    gcs_uri=gcs_uri,
                    cached_content=cached_content,
                    chunk=chunk,
                )
                for chunk in chunks
            ],
            return_exceptions=True,
        )

        enriched_chunks: list[ParagraphData] = []
        errors: list[tuple[int, Exception]] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append((i, result))
                self.logger.error(
                    f"ERROR on chunk {chunks[i].object_id} for document {chunks[i].file_name} on page {chunks[i].pages[0].page_number}: {result}")
                # return the original chunk without enrichment
                enriched_chunks.append(chunks[i])
            else:
                enriched_chunks.append(result)

        if errors:
            self.logger.info(f"\n{len(errors)}/{len(chunks)} chunks failed.")

        # sort the enriched_chunks by object id to make sure that the structure remain the same
        return sorted(enriched_chunks, key=lambda x: x.object_id)

    async def enrich_chunks_pipeline(self,
                                     *,
                                     chunks: List[ParagraphData],
                                     ttl_minutes: int = None) -> List[ParagraphData]:
        """pipeline for processing chunks using context caching process"""

        # STEP 1 - CREATE THE URI
        gcs_uri, blob = self._upload_pdf_file_to_memory()

        # STEP 2 - COUNT PDF TOKENS AND DETERMINE THE CACHE TTL
        num_tokens = await self._count_tokens(gcs_uri=gcs_uri)

        # Estimate TTL from chunk count when the existing token threshold is met.
        if ttl_minutes is None and num_tokens > 4096:
            ttl_minutes = self.calculate_ttl_minutes(num_chunks=len(chunks))

        # STEP 3 - CREATE THE PDF CACHE IF TTL IS AVAILABLE
        cached_content = await self.create_pdf_cache(ttl_minutes=ttl_minutes, gcs_uri=gcs_uri)

        try:
            # STEP 4 - UPDATE THE KNOWLEDGE BASE FILES (DOCUMENT LEVEL)
            await self.update_knowledgebases(gcs_uri=gcs_uri, cached_content=cached_content)

            # STEP 5 - TAG THE DOCUMENT AND SPREAD ITS TAGS TO ALL OF ITS CHUNKS (DOCUMENT LEVEL)
            document_tags = await self.tag_document(gcs_uri=gcs_uri, cached_content=cached_content)
            for chunk in chunks:
                chunk.document_classification = document_tags.get("document_classification")
                chunk.evidence_scale = document_tags.get("evidence_scale")

            # STEP 6 - ENRICH THE CHUNKS (CHUNK LEVEL)
            return await self.enrich_chunks(chunks=chunks, gcs_uri=gcs_uri, cached_content=cached_content)

        finally:
            # STEP 7 - DELETE THE CACHE IF ONE WAS CREATED
            if cached_content:
                await self.gemini_client.aio.caches.delete(name=cached_content.name)
                self.logger.info(f"Cache deleted for {self.blob_path_pdf}.")


# ----- Agent 5: Sentence Splitter Agent with Context Caching
# Replace the existing SentenceSplitAgent in assistant_agents.py with this version

@dataclass
class ContextSentenceSplitAgent:
    """Split parser JSON objects into individual sentences using Gemini with context-cached PDF.

    Uses genai.Client directly (not pydantic-ai) to support context caching.
    The PDF is cached once, then each batch is sent as a prompt referencing the cache.
    Response is parsed into AgenticResponse.
    """

    model_name: str
    project_id: str
    google_provider_region: str
    caching_location: str
    path_to_system_prompt: Path
    path_to_batch_prompt: Path
    path_to_document_prompt: Path
    max_concurrent: int = 5
    temperature: float = 0.0
    thinking_budget: int = 1024

    # PDF source (set before calling run)
    bucket_name_pdf: Optional[str] = None
    blob_path_pdf: Optional[str] = None

    def __post_init__(self):
        self.system_prompt = Path(self.path_to_system_prompt).read_text(encoding="utf-8")
        self.batch_prompt = Path(self.path_to_batch_prompt).read_text(encoding="utf-8")
        self.document_prompt = Path(self.path_to_document_prompt).read_text(encoding="utf-8")
        self.logger = setup_logger(self.__class__.__name__)
        self.gemini_client = self._initialize_client()
        self.storage_client = storage.Client()

    def _initialize_client(self):
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.caching_location,
            http_options=HttpOptions(
                base_url=f"https://{self.caching_location}-aiplatform.googleapis.com"),
        )
        api = client._api_client
        logger.info("httpx clients: sync_hooks=%s async_hooks=%s",
                    getattr(api, "_httpx_client", None) and api._httpx_client.event_hooks,
                    getattr(api, "_async_httpx_client", None) and api._async_httpx_client.event_hooks)
        logger.info("GoogleProvider ready (hooks registered on sync + async clients)")
        return client

    def _get_system_prompt(self) -> str:
        return self.system_prompt

    def _build_safety_settings(self):
        return [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

    def _get_gcs_uri(self) -> str:
        """Build the GCS URI for the PDF file."""
        return f"gs://{self.bucket_name_pdf}/{self.blob_path_pdf}"

    async def _count_tokens(self, gcs_uri: str) -> int:
        """Count tokens in the PDF to verify context caching eligibility (min 1024)."""
        response = await self.gemini_client.aio.models.count_tokens(
            model=self.model_name,
            contents=types.Content(
                role="user",
                parts=[types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf")],
            ),
        )
        self.logger.info(f"PDF token count: {response.total_tokens}")
        return response.total_tokens

    def calculate_ttl_minutes(self, num_batches: int) -> int:
        """Estimate cache TTL based on batch count and concurrency.

        Each batch takes ~15-30s for the LLM to process (sentence splitting is heavier
        than enrichment). We estimate conservatively and add a buffer.
        """
        avg_seconds_per_batch = 50.0
        estimated_seconds = (num_batches / self.max_concurrent) * avg_seconds_per_batch
        buffered_seconds = estimated_seconds * 1.5
        ttl_minutes = max(7, min(120, int(buffered_seconds / 60) + 1))
        self.logger.info(f"Cache TTL: {ttl_minutes} minutes for {num_batches} batches")
        return ttl_minutes

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    async def _create_cache(self, gcs_uri: str, ttl_minutes: int) -> Any:
        """Create a context cache with the PDF baked in."""
        cached_content = await self.gemini_client.aio.caches.create(
            model=self.model_name,
            config=types.CreateCachedContentConfig(
                display_name=f"sentence-split-cache-{Path(self.blob_path_pdf).stem}",
                system_instruction=self._get_system_prompt(),
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf"),
                            types.Part.from_text(text=self.document_prompt),
                        ],
                    )
                ],
                ttl=f"{ttl_minutes * 60}s",
            ),
        )
        self.logger.info(f"Cache created: {cached_content.name}, expires: {cached_content.expire_time}")
        return cached_content

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    async def _process_batch(self,
                             *,
                             semaphore: asyncio.Semaphore,
                             cached_content: Any,
                             gcs_uri: str,
                             batch: list[dict],
                             batch_idx: int) -> AgenticResponse:
        """Process a single batch using the cached PDF context."""
        prompt = self.batch_prompt.format(batch=batch)

        async with semaphore:
            if cached_content is not None:
                response = await self.gemini_client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cached_content.name,
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                        safety_settings=self._build_safety_settings(),
                        temperature=self.temperature,
                        response_mime_type='application/json',
                        response_schema=SimpleAgenticResponse,
                    ),
                )
            else:
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=gcs_uri, mime_type="application/pdf"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ]
                response = await self.gemini_client.aio.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._get_system_prompt(),
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                        safety_settings=self._build_safety_settings(),
                        temperature=self.temperature,
                        response_mime_type='application/json',
                        response_schema=SimpleAgenticResponse,
                    ),
                )
        self.logger.info(f"Batch {batch_idx} processed successfully")

        try:
            # simple_response = SimpleAgenticResponse.model_validate_json(raw_text)

            simple_response: SimpleAgenticResponse = response.parsed

            complex_paragraphs = []
            for p in simple_response.paragraphs:
                complex_paragraphs.append(ParagraphData(
                    object_id=p.object_id,
                    file_name=p.file_name,
                    content=p.content,
                    type=[ParagraphType(type=t) for t in p.type],
                    pages=[ParagraphPageNumber(page_number=page) for page in p.pages],
                    location=[ParagraphLocation(locations=p.locations)]
                    # location=[ParagraphLocation(locations=loc) for loc in p.locations]
                ))
            self.logger.info(f"Batch {batch_idx} validated and mapped to complex schema successfully")

            return AgenticResponse(paragraphs=complex_paragraphs)
        except (ValidationError, ValueError) as e:
            self.logger.warning(f"Batch {batch_idx} JSON validation failed: {e}")
            raise ModelValidateSchemaError(f"Failed to validate JSON schema for batch {batch_idx}") from e

    async def run(self, batches: list[list[dict]]) -> list[ParagraphData]:
        gcs_uri = self._get_gcs_uri()
        self.logger.info(f"PDF URI: {gcs_uri}")
        num_tokens = await self._count_tokens(gcs_uri)
        cached_content = None
        if num_tokens >= 1024:
            ttl_minutes = self.calculate_ttl_minutes(num_batches=len(batches))
            cached_content = await self._create_cache(gcs_uri, ttl_minutes)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        try:
            tasks = [
                self._process_batch(
                    semaphore=semaphore,
                    cached_content=cached_content,
                    gcs_uri=gcs_uri,
                    batch=batch,
                    batch_idx=k,
                )
                for k, batch in enumerate(batches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_sentences: list[ParagraphData] = []
            for k, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch {k} failed: {result}")
                else:
                    # result is an AgenticResponse — extract its paragraphs
                    all_sentences.extend(result.paragraphs)

            self.logger.info(f"Sentence splitting complete: {len(all_sentences)} sentences from {len(batches)} batches")
            return all_sentences

        finally:
            if cached_content:
                await self.gemini_client.aio.caches.delete(name=cached_content.name)
                self.logger.info("Sentence split cache deleted")
