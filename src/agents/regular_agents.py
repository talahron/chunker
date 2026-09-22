from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

from google import genai
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    HttpOptions,
    SafetySettingDict,
    ThinkingConfigDict,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.exceptions import AgenticChunkingError, _is_retryable
from src.logger import setup_logger
from src.pydantic_models import AgenticResponse, SubChunks
from pathlib import Path
logger = setup_logger(__name__)

def initialize_google_provider(*, project: str, location: str) -> GoogleProvider:
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(
            base_url="https://aiplatform.googleapis.com",
        ),
    )
    return GoogleProvider(client=client)

# ----- Agent 1: split large chunks
@dataclass
class LargeChunksSplitAgent:
    """split large chunks based on agentic chunking strategy"""
    model_name: str
    project_id: str
    google_provider_region: str
    path_to_system_prompt: Path

    def __post_init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.shared_vertex_provider = initialize_google_provider(project=self.project_id,
                                                                 location=self.google_provider_region)
        self._extract_agent()
        self.token_refresh_lock = asyncio.Lock()  # Lock for safe token refresh

    def _extract_agent(self) -> None:
        model = GoogleModel(self.model_name, provider=self.shared_vertex_provider)
        self.agent = Agent(model, system_prompt=self._get_system_prompt(), output_type=SubChunks)

    def _get_system_prompt(self) -> str:
        return Path(self.path_to_system_prompt).read_text(encoding="utf-8")

    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    async def run(self, text: str) -> List[str]:
        """
        wait_random_exponential(multiplier=1, max=60) means that it will
        Retry “Randomly wait up to 2^x * 1 seconds between each retry until the
        range reaches 60 seconds, then randomly up to 60 seconds afterwards.
        """
        async with self.token_refresh_lock:  # Lock to ensure single refresh at a time
            try:
                result = await self.agent.run([text])
                resp: SubChunks = result.output
                paragraphs = [seg.content.strip() for seg in (resp.sub_chunks or []) if seg.content.strip()]
                return paragraphs

            except AgenticChunkingError as exc:
                self.logger.error(f"Agent Chunker failed - running simple chunker - {exc}")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100)
                paragraphs = text_splitter.split_text(text)
                return paragraphs


# ----- Agent 2: text summarizer
@dataclass
class SummarizerAgent:

    """Summarizer agent"""

    model_name: str
    project_id: str
    google_provider_region: str
    path_to_system_prompt: Path

    def __post_init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.shared_vertex_provider = initialize_google_provider(project=self.project_id,
                                                                 location=self.google_provider_region)
        self._extract_agent()
        self.token_refresh_lock = asyncio.Lock()  # Lock for safe token refresh

    def _extract_agent(self) -> None:
        model = GoogleModel(self.model_name, provider=self.shared_vertex_provider)
        self.agent = Agent(model, system_prompt=self._get_system_prompt())

    def _get_system_prompt(self) -> str:
        return Path(self.path_to_system_prompt).read_text(encoding="utf-8")

    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    async def run(self, text: str) -> str:
        """
        wait_random_exponential(multiplier=1, max=60) means that it will
        Retry “Randomly wait up to 2^x * 1 seconds between each retry until the
        range reaches 60 seconds, then randomly up to 60 seconds afterwards.
        """
        async with self.token_refresh_lock:  # Lock to ensure single refresh at a time
            resp = await self.agent.run(text)
            return resp.output


# ----- Agent 3: Agentic agent text splitter
@dataclass
class AgenticTextAgent:

    """Agentic text agent"""

    model_name: str
    project_id: str
    google_provider_region: str
    path_to_system_prompt: Path
    temperature: float = 0.0

    def __post_init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self._initialize_google_provider()
        self._extract_agent()

    def _initialize_google_provider(self):
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.google_provider_region,
            http_options=HttpOptions(
                base_url="https://aiplatform.googleapis.com",
            ),
        )
        self.vertex_provider = GoogleProvider(client=client)
        api = client._api_client
        logger.info("httpx clients: sync_hooks=%s async_hooks=%s",
                    getattr(api, "_httpx_client", None) and api._httpx_client.event_hooks,
                    getattr(api, "_async_httpx_client", None) and api._async_httpx_client.event_hooks)
        logger.info("GoogleProvider ready (hooks registered on sync + async clients)")

    def _extract_agent(self) -> None:
        model = GoogleModel(self.model_name, provider=self.vertex_provider)
        self.agent = Agent(model, system_prompt=self._get_system_prompt(), output_type=AgenticResponse)

    def _get_system_prompt(self) -> str:
        return Path(self.path_to_system_prompt).read_text(encoding="utf-8")

    def _init_model_settings(self, thinking_budget: int = 1024):
        return GoogleModelSettings(
            temperature=self.temperature,
            google_thinking_config=ThinkingConfigDict(thinking_budget=thinking_budget),
            google_safety_settings=[
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                )
            ]
        )

    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),  # Add maximum attempts
           retry=retry_if_exception(_is_retryable),
           reraise=True)  # Re-raise exceptions
    async def run(self, text: str) -> AgenticResponse:
        """
        wait_random_exponential(multiplier=1, max=60) means that it will
        Retry “Randomly wait up to 2^x * 1 seconds between each retry until the
        range reaches 60 seconds, then randomly up to 60 seconds afterwards.
        """
        resp = await self.agent.run(text, model_settings=self._init_model_settings())
        return resp.output


# ----- Agent 4: Sentence Splitter Agent
# Append this to the end of assistant_agents.py
@dataclass
class SentenceSplitAgent:
    """Split parser JSON objects into individual sentences using an LLM agent.

    Similar to AgenticTextAgent but with a prompt focused on sentence-level
    splitting rather than paragraph aggregation. Content is cleaned of stray
    punctuation before being sent to the agent.
    """

    model_name: str
    project_id: str
    google_provider_region: str
    path_to_system_prompt: Path
    temperature: float = 0.0

    def __post_init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self._initialize_google_provider()
        self._extract_agent()

    def _initialize_google_provider(self):
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.google_provider_region,
            http_options=HttpOptions(base_url="https://aiplatform.googleapis.com"),
        )
        api = client._api_client
        logger.info("httpx clients: sync_hooks=%s async_hooks=%s",
                    getattr(api, "_httpx_client", None) and api._httpx_client.event_hooks,
                    getattr(api, "_async_httpx_client", None) and api._async_httpx_client.event_hooks)
        logger.info("GoogleProvider ready (hooks registered on sync + async clients)")
        self.vertex_provider = GoogleProvider(client=client)

    def _extract_agent(self) -> None:
        model = GoogleModel(self.model_name, provider=self.vertex_provider)
        self.agent = Agent(model, system_prompt=self._get_system_prompt(), output_type=AgenticResponse)

    def _get_system_prompt(self) -> str:
        return Path(self.path_to_system_prompt).read_text(encoding="utf-8")

    def _init_model_settings(self, thinking_budget: int = 1024):
        return GoogleModelSettings(
            temperature=self.temperature,
            google_thinking_config=ThinkingConfigDict(thinking_budget=thinking_budget),
            google_safety_settings=[
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                SafetySettingDict(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                )
            ]
        )


    @retry(wait=wait_random_exponential(multiplier=1, max=60),
           stop=stop_after_attempt(3),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    async def run(self, text: str) -> AgenticResponse:
        resp = await self.agent.run(text, model_settings=self._init_model_settings())
        return resp.output
