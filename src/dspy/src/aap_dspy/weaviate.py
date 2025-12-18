from collections.abc import Sequence
from typing import Tuple

from aap_core.prompt_enhancer import BaseRAGPromptEnhancer
from aap_core.types import AgentMessage, ContentType
from pydantic import Field
from dspy.retrieve.weaviate_rm import WeaviateRM
import dspy


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
    rm: WeaviateRM = Field(..., description="The Weaviate retriever model object")

    def _search(self, query: str, **kwargs) -> Sequence[Tuple[ContentType, str]]:
        context = dspy.Retrieve(**kwargs)(query).passages
        return [("text", c) for c in context]

    def _format(
        self, message: AgentMessage, data: Sequence[Tuple[ContentType, str]]
    ) -> AgentMessage:
        context = "\n".join([content for _, content in data])
        # TODO: handle multimodal data
        message.query = self.format.format(query=message.query, context=context)
        return message
