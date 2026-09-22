from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from src.exceptions import ObjectJsonDownloadError, BatchPredictionError, SummarizerAgentError
from src.pydantic_models import (
    DocumentObject, Chunk, ChunkMetaData, ObjectMetaData, ParagraphData,
    ParagraphType, ParagraphPageNumber, ParagraphLocation,
)
from langchain_google_vertexai import VertexAIEmbeddings

class ChunkerUtils:
    """Base class containing shared methods for chunking operations."""

    def __init__(self):
        self.logger = None
        self.storage = None
        self.input_bucket = None
        self.file_name = None
        self.base_name = None
        self.increment = None
        self.sentence_agent = None
        self.document: List[DocumentObject] = []

    async def _download_parser_json(self):
        """Download and parse JSON from GCS bucket."""
        try:
            self.logger.info(f"Downloading parser json object - {self.file_name}")
            # Download parsed JSON from GCS (expects gs://<bucket>/<base>/main_json.json)
            parsed_json, self.base_name = await self.storage.download_json(
                self.input_bucket,
                self.file_name,
            )
            document_objects = parsed_json["document"]
            self.document: List[DocumentObject] = [
                DocumentObject(**obj) for obj in document_objects]
            self.logger.info(f'Successfully download parser json. Document name - {self.file_name}')
        except ObjectJsonDownloadError as e:
            self.logger.error(f'Failed to download parser json. Document name - {self.file_name}')
            raise

    @property
    def _convert_parser_output_to_dict(self):
        """convert the self.document which is list[DocumentObject] into list[Dict[str, Any]]"""
        if self.document: # the object is not empty
            self.logger.info("Converting the parser document from format of DocumentObject to dict")
            return [doc.model_dump() for doc in self.document]
        else:
            self.logger.info("Unable to convert the parser document from format of DocumentObject to dict")
            return []

    def _batch_prediction(self, parser_object_convert_to_dict: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """
        create batch of text based on the length of the parser object
        the process follows these steps:
        Next, create dictionary that holds each object and its chars count
        Next, calculate their cumulative count and create the relevant groups based on the specified increment
        :return: A batch of text divided by the increment value into groups
        """
        """creating a batch of parser json objects"""
        # first, need to make sure that the parser_object_convert_to_dict is not an empty list
        if not parser_object_convert_to_dict:
            self.logger.debug("parser object is an empty list - stopped executing the function")
            return []
        try:
            self.logger.info("Start creating batch of text to process")
            dictionary_batch = {k: len(str(doc_obj)) for k, doc_obj in enumerate(parser_object_convert_to_dict)}
            num_chars_per_doc_object = list(dictionary_batch.values())
            cumulative_sum = np.cumsum(num_chars_per_doc_object)
            groups_list = []
            for i in range(0, int(cumulative_sum.max()) + self.increment, self.increment):
                group = np.where((cumulative_sum >= i) & (cumulative_sum < i + self.increment))
                if len(group) > 0:
                    groups_list.append(group[0])
            batch_texts = []
            for group_values in groups_list:
                if len(group_values)>0:
                    batch_texts.append(list(np.array(parser_object_convert_to_dict)[group_values]))
            self.logger.info("Successfully created batch of text to process")
            return batch_texts
        except BatchPredictionError as e:
            self.logger.debug(f"cannot create batch of text to process - stopped executing the function - {e}")
            raise

    async def _split_into_sentences(self) -> List[ParagraphData]:
        """
        LLM-based sentence splitting with PDF context cache.
        Shared by all semantic chunkers. Requires self.sentence_agent to be
        set by the concrete chunker's setup().
        """
        converted = self._convert_parser_output_to_dict
        batches = self._batch_prediction(converted)
        sentences = await self.sentence_agent.run(batches)
        self.logger.info(f"Sentence splitting produced {len(sentences)} sentences")
        return sentences

    async def _build_content_and_metadata(self,
                                          agentic_paragraphs: list[ParagraphData],
                                          contents: list[str] = None,
                                          metadata: list[ChunkMetaData] = None) -> Tuple[List[str], List[ChunkMetaData]]:
        if contents is None:
            contents: list[str] = []
        if metadata is None:
            metadata: list[ChunkMetaData] = []
        # loop over the entire agentic paragraphs to create the content and the metadata
        self.logger.info("Start building content and metadata")
        for k, chunk in enumerate(agentic_paragraphs):
            self.logger.debug(f"Running building content and metadata for chunk number - {k} / {len(agentic_paragraphs)}")
            contents.append(chunk.content if chunk.content is not None else "")
            # flatten the pages to a string object
            pages = str(sorted({page.page_number for page in chunk.pages if page.page_number is not None}))
            # save the file name in separate value
            file_name = chunk.file_name if chunk.file_name is not None else ""
            metadata.append(
                ChunkMetaData(
                    file_name=file_name,
                    chunk_number=chunk.object_id,
                    pages=pages,
                )
            )
        metadata_json_schema = [m.model_dump() for m in metadata]
        self.logger.info(f"Chunks and Metadata build complete successfully. number of chunks {len(contents)}.")
        return contents, metadata_json_schema
