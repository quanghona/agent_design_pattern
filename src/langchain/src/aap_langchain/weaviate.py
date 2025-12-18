from typing import Sequence, Tuple

from aap_core.prompt_enhancer import BaseRAGPromptEnhancer
from aap_core.types import AgentMessage, ContentType
from langchain_weaviate import WeaviateVectorStore
from pydantic import Field, field_validator


class WeaviatePromptEnhancer(BaseRAGPromptEnhancer):
    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    vector_store: WeaviateVectorStore = Field(
        ..., description="The vector store object"
    )

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{context}" not in v:
            raise ValueError("The format must contain at least {query} and {context}")
        return v

    def _search(self, query: str, **kwargs) -> Sequence[Tuple[ContentType, str]]:
        docs = self.vector_store.similarity_search(query=query, **kwargs)
        results = []
        for doc in docs:
            results.append(("text", doc.page_content))
        # TODO: parse other content types
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
