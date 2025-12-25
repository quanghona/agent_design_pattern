import abc
from collections.abc import Callable

from pydantic import Field, field_validator

from aap_core.retriever import BaseRetriever

# import toon_format
from .types import AgentMessage, BaseChain


class BasePromptAugmenter(BaseChain):
    """A base class to enhance / rewrite the prompt.

    There are two types of prompt enhancement:
    - Data enhancement: Give more context to the prompt by adding external data, either by using files (CSV, JSON, Markdown, etc.), database (SQL,...) or more advanced techniques like RAG, web search.
    - Structure enhancement: Rewrite / refine the prompt partially or entirely.
    """

    # TODO: handle case where single augmenter contains multiple retrievers
    loop: int | Callable[[AgentMessage], bool] | None = Field(
        default=None,
        description="The loop, either by number of times or by stop condition",
    )
    retriever: BaseRetriever | None = Field(default=None, description="The retriever")

    async def acall(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    @abc.abstractmethod
    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    def call(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if self.retriever:
            message = self.retriever(message, **kwargs)
        return self.augment(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if isinstance(self.loop, int):
            for _ in range(self.loop):
                message = self.call(message, **kwargs)
            return message
        elif callable(self.loop):
            while self.loop(message):
                message = self.call(message, **kwargs)
            return message
        else:
            return self.call(message, **kwargs)


class IdentityPromptAugmenter(BasePromptAugmenter):
    """A prompt enhancer that does nothing.
    This serves as a default prompt enhancer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = None
        self.retriever = None

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message


class SimplePromptAugmenter(BasePromptAugmenter):
    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    data_key: str = Field(
        default="context.data", description="The key to the data in the message"
    )

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{data}" not in v:
            raise ValueError("The format must contain at least {query} and {data}")
        return v

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("task_response_key must start with 'context.'")
        return v

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if message.context is None:
            raise ValueError("Message context is None")
        elif self.data_key not in message.context:
            raise ValueError(f"Message context does not contain {self.data_key}")
        context_key = self.data_key.replace("context.", "")
        message.query = self.format.format(
            query=message.query, data=message.context[context_key], **kwargs
        )
        return message


class GEPAPromptAugmenter(BasePromptAugmenter):
    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        raise NotImplementedError
