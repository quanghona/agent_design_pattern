import abc
from collections.abc import Callable
import json
from typing import Any, ClassVar, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    query: str = Field(...,description="The prompt consumed by agent's LLM. Note that this is a user prompt")
    origin: Optional[str] = Field(None, description="The agent that send this message")
    response: Optional[str] = Field(None, description="The response from the agent's LLM")
    responses: Optional[List[Tuple[str, str]]] = Field(None, description="""
        If an agent generate multiple responses, either by same or different subagents, all of them will be stored here.
        In each response tuple, the first one should be agent name or index, and second is the response""")
    context: Optional[dict] = Field(None, description="""
        Agent additional material, which probably is the output of other agent or user entered.
        There are various types of context produced by user and other agents.
        The prompt that comsume this context need to explicitly know the format of this context.""")
    execution_result: Optional[Literal["success", "error"]] = Field(None, description="The execution result of the agent. Can be success or error")
    error_message: Optional[str] = Field(None, description="The error message if the execution result is error.")
    # media: Optional[List[str]] = Field(None, description="The additional media content. Can be image, video, audio, etc.")
    # media_type: Optional[List[str]] = Field(None, description="The media type associated with the media.")

    def flatten_dict(self, d: dict, parent_key: str = '', sep='_'):
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

    def to_dict(self) -> dict[str, Any]:
        msg_json =  self.model_dump(exclude_none=True, exclude={"context"})
        if self.context:
            context_dict = self.flatten_dict(self.context, parent_key="context")
            for k, v in context_dict.items():
                msg_json[k] = v
        return msg_json

    def dump_json(self) -> str:
        msg_dict = self.to_dict()
        return json.dumps(msg_dict)


class LLMChain(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

    async def ainvoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)


class BaseAgent(abc.ABC):
    _object_count: ClassVar[int] = 0
    """An interface for agent implementations."""
    def __init__(self, state_change_callback: Callable[[str], None] = None, name: str = None, **kwargs):
        super().__init__()
        BaseAgent._object_count += 1
        self._state = "idle"
        self.state_change_callback = state_change_callback
        self.name = name if name else f"{type(self).__name__}_{BaseAgent._object_count}"
        # TODO: maybe follow A2A's agent card and agent skill
        # self.card = None
        # self.skills = None

    @abc.abstractmethod
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        """The execution logic of the agent.
        Depends on the detail implementation by subclass, agent can calls tools or RAG or MCP during execution.
        And agent may or may not utilize LLM, depends on the implementation. A lot of agents are pure orchestration without LLM in it.
        During execution, the implementation should update the state of the agent
        and/or push notification to target to update the progress.
        There are 3 types of agent:
        - Remote agent: agent that is running on other server.
        Example: For Google's A2A, we will act as a client agent and communicate with agent using A2A protocol and getback result
        - Local agent: self-defined agent that is running on a this machine.
        - Orchestration agent: itself is a agent, which can perform what normal agent can do, and orchestrate other agents to complete the provided task.

        Args:
            message (AgentMessage): The message to execute the agent with.
            **kwargs: Additional keyword arguments to pass to the LLM generation method.

        Returns:
            AgentMessage: The response from the agent.
        """
        return message

    async def aexecute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.execute(message, **kwargs)

    @property
    def state(self):
        """Get the current state of the agent."""
        return self._state

    def _set_state(self, state: str):
        self._state = state
        if self.state_change_callback:
            self.state_change_callback(state)
