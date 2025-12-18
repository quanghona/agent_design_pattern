from collections.abc import Sequence
from typing import Tuple

from aap_core.prompt_enhancer import BaseRAGPromptEnhancer
from aap_core.types import AgentMessage, ContentType
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from pydantic import Field, PrivateAttr, field_validator


class WeaviatePromptEnhancer(BaseRAGPromptEnhancer):
    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    index: VectorStoreIndex = Field(..., description="The vector store index object")

    _retriever: BaseRetriever | None = PrivateAttr(None)

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{context}" not in v:
            raise ValueError("The format must contain at least {query} and {context}")
        return v

    def _search(self, query: str, **kwargs) -> Sequence[Tuple[ContentType, str]]:
        results = []
        if self._retriever:
            nodes = self._retriever.retrieve(query)
            for node in nodes:
                results.append(("text", node.get_content()))
            # TODO: parse other content types
        else:
            raise RuntimeError(
                "Retriever not built yet. use .build_retriever method to build the retriever from index before searching"
            )
        return results

    def _format(
        self,
        message: AgentMessage,
        data: Sequence[Tuple[ContentType, str]],
        separator: str = "",
        **kwargs,
    ) -> AgentMessage:
        context = separator.join([content for _, content in data])
        # TODO: handle multimodal data
        message.query = self.format.format(query=message.query, context=context)
        return message

    def build_retriever(self, **kwargs) -> "WeaviatePromptEnhancer":
        """
        Builds a retriever from the index.

        Parameters:
            **kwargs: Additional keyword arguments passed to `VectorStoreIndex.as_retriever`
            Refer to: https://developers.llamaindex.ai/python/framework-api-reference/indices/document_summary/?h=vectorstoreindex#llama_index.core.indices.DocumentSummaryIndex.as_retriever
            for the available options

        Returns:
            self: The WeaviatePromptEnhancer instance
        """
        self._retriever = self.index.as_retriever(**kwargs)
        return self
