import abc
from collections.abc import Callable
from typing import Optional
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    query: str = Field(description="The prompt consumed by agent's LLM. Note that this is a user prompt")
    origin: str = Field(description="The agent that send this message", default="")
    response: Optional[str] = Field(description="The response from the agent's LLM", default="")
    artifact: Optional[dict] = Field(description="""
        Agent additional material, which probably is the output of other agent or user entered.
        There are various types of artifact produced by user and other agents.
        The prompt that comsume this artifact need to explicitly know the format of this artifact.""")
    execution_result: Optional[str] = Field(description="The execution result of the agent. Can be success or error", default="success")
    error_message: Optional[str] = Field(description="The error message if the execution result is error.", default="")

    def to_dict(self):
        return {
            "query": self.query,
            "origin": self.origin,
            "response": self.response,
            "execution_result": self.execution_result,
            "error_message": self.error_message,
            "artifact": self.artifact.__dict__
        }


class LLMChain(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

class IAgent(abc.ABC):
    _object_count = 0
    """An interface for agent implementations."""
    def __init__(self, state_change_callback: Callable[[str], None] = None, name: str = None, **kwargs):
        super().__init__()
        IAgent._object_count += 1
        self._state = "idle"
        self.tools = []     # internal tools, RAG or MCP
        self.state_change_callback = state_change_callback
        self.name = name if name else f"{type(self).__name__}_{IAgent._object_count}"

    @abc.abstractmethod
    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        """The execution logic of the agent.
        Depends on the detail implementation by subclass, agent can calls tools or RAG or MCP during execution.
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
