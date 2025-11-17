import abc
from collections.abc import Callable
from typing import Optional

from a2a.types import AgentCard
from pydantic import BaseModel, Field, PrivateAttr

from .types import AgentMessage


class BaseAgent(abc.ABC, BaseModel):
    """An interface for agent implementations."""

    _state: str = PrivateAttr("idle")

    card: AgentCard = Field(
        ...,
        description="""
        A self-describing manifest for an agent.
        It provides essential metadata including the agent's identity, capabilities, skills, supported communication methods, and security requirements.""",
    )
    state_change_callback: Optional[Callable[[str], None] | None] = Field(
        default=None,
        description="The callback function to update the state of the agent.",
    )

    def __init__(
        self,
        card: AgentCard,
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__()
        self._state = "idle"
        self.state_change_callback = state_change_callback
        self.card = card

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
