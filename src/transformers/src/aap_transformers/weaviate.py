from collections.abc import Callable, Mapping, Sequence
from typing import Any, Tuple

from aap_core.prompt_enhancer import BaseRAGPromptEnhancer
from aap_core.types import AgentMessage, ContentType
from pydantic import Field
from weaviate.collections.collection.sync import Collection


class WeaviatePromptEnhancer(BaseRAGPromptEnhancer):
    """
    Regarding the implementation of WeaviatePromptEnhancer for transformer, there is no in-house support like langchain or llamaindex.
    We need to access the weaviate object and build the logic ourself.
    """

    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    collection: Collection = Field(..., description="The Weaviate collection object")
    extract_func: Callable[[Mapping[str, Any]], Tuple[ContentType, str]]

    def _search(self, query: str, **kwargs) -> Sequence[Tuple[ContentType, str]]:
        context = self.collection.query.near_text(query=query)
        # TODO: handle multimodal search
        results = []

        for obj in context.objects:
            results.append(self.extract_func(obj.properties))

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
