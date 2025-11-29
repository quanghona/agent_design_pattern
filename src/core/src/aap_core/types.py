import abc
import json
from typing import Any, Dict, List, Literal, Tuple, TypeVar

from pydantic import BaseModel, Field


ChainMessage = TypeVar("ChainMessage")
ChainResponse = TypeVar("ChainResponse")
AgentResponse = Tuple[str, str]
ContentType = Literal["text", "image", "audio", "video", "document"]


class AgentMessage(BaseModel):
    query: str = Field(
        ...,
        description="""The user query consumed by agent's LLM.
        In this message class, we don't store the system prompt and user template.
        Because each agent have their own system prompt and user template, which is not shared between agents.""",
    )
    query_media: List[Tuple[ContentType, str]] | None = Field(
        default=None, description="The media content associated with the query."
    )
    origin: str | None = Field(
        default=None, description="The agent that send this message"
    )
    responses: List[AgentResponse] = Field(
        default=[],
        description="""
        If an agent generate multiple responses, either by same or different subagents, all of them will be stored here.
        In each response tuple, the first one should be agent name or index, and second is the response""",
    )
    context: dict | None = Field(
        default=None,
        description="""
        Agent additional material, which probably is the output of other agent or user entered.
        There are various types of context produced by user and other agents.
        The prompt that comsume this context need to explicitly know the format of this context.""",
    )
    execution_result: Literal["success", "error"] | None = Field(
        default=None,
        description="The execution result of the agent. Can be success or error",
    )
    error_message: str | None = Field(
        default=None, description="The error message if the execution result is error."
    )
    media: List[Tuple[ContentType, str]] | None = Field(
        default=None,
        description="The additional media content. Can be image, video, audio, etc.",
    )

    def flatten_dict(self, d: dict, parent_key: str = "", sep="_") -> Dict[str, Any]:
        """
        Flattens a nested dictionary into a single-level dictionary.

        Args:
            d (dict): The dictionary to flatten.
            parent_key (str): The prefix for keys in the flattened dictionary.
            sep (str): The separator used to join parent and child keys.

        Returns:
            dict: The flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the AgentMessage object into a dictionary format.

        Returns a dictionary that contains all the information in the AgentMessage object.
        If the context is not None, it will be flattened and added to the returned dictionary.
        All the data in context will have prefix of 'context_' by default.
        For example, if the context is {'a': 1, 'b': 2}, the returned dictionary will contains {'context_a': 1, 'context_b': 2}

        Returns:
            dict: A dictionary that contains all the information in the AgentMessage object.
        """
        msg_json = self.model_dump(exclude_none=True, exclude={"context"})
        if self.context:
            context_dict = self.flatten_dict(self.context, parent_key="context")
            for k, v in context_dict.items():
                msg_json[k] = v
        return msg_json

    def dump_json(self) -> str:
        """
        Converts the AgentMessage object into a JSON string.

        Returns:
            str: A JSON string that contains all the information in the AgentMessage object.
        """
        msg_dict = self.to_dict()
        return json.dumps(msg_dict)


class BaseChain(abc.ABC, BaseModel):
    @abc.abstractmethod
    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass
