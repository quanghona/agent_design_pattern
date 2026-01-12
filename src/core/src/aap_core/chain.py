import abc
from collections.abc import Sequence
from typing import Callable, Generic, List, Tuple

from pydantic import Field, PrivateAttr

from .guardrail import BaseGuardRail, PassGuardRail
from .prompt_augmenter import (
    BasePromptAugmenter,
    IdentityPromptAugmenter,
)
from .types import AgentMessage, BaseChain, ChainMessage, ChainResponse


class BaseLLMChain(BaseChain):
    """
    Base class for LLM chains.
    """

    name: str = Field(
        "chain",
        description="The name of the chain. Should be same at agent who hold this chain for easy to operate.",
    )

    @abc.abstractmethod
    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

    async def ainvoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)


class BaseCausalMultiTurnsChain(BaseLLMChain, Generic[ChainMessage, ChainResponse]):
    """
    Base class for multi-turns chains.
    """

    _last_response_as_context: str | None = PrivateAttr(default=None)
    include_history: int = Field(
        default=0, description="number of history message turns to include"
    )
    store_immediate_steps: bool = Field(
        default=False, description="store intermediate steps output"
    )

    def final_response_as_context(self, key: str) -> None:
        self._last_response_as_context = key.replace("context_", "")

    @abc.abstractmethod
    def _prepare_conversation(self, message: AgentMessage) -> List[ChainMessage]:
        pass

    @abc.abstractmethod
    def _generate_response(
        self, conversation: List[ChainMessage], **kwargs
    ) -> Tuple[List[ChainMessage], ChainResponse, bool]:
        pass

    @abc.abstractmethod
    def _process_tools(
        self,
        conversation: List[ChainMessage],
        response: ChainResponse,
    ) -> List[ChainMessage]:
        pass

    @abc.abstractmethod
    def _append_responses(
        self, message: AgentMessage, conversation: List[ChainMessage]
    ) -> AgentMessage:
        pass

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Template method
        conversation = self._prepare_conversation(message)
        conversation, response, has_tool = self._generate_response(
            conversation, **kwargs
        )
        if has_tool:
            conversation = self._process_tools(conversation, response)
            conversation, _, _ = self._generate_response(conversation, **kwargs)

        message = self._append_responses(message, conversation)

        if self._last_response_as_context is not None:
            if message.context is None:
                message.context = {}
            message.context[self._last_response_as_context] = message.responses.pop()[1]

        message.origin = self.name
        message.execution_result = "success"
        return message


class TypicalLLMChain(BaseLLMChain):
    """
    A LLM chain that consists of all components in a typical response generation.
    The chain has 4 steps:
    - input guardrail: validate the input message
    - prompt enhancement: add more context to the prompt or rewrite / refine the prompt
    - generate: generate the response. This steps should also handle the tool calling procedure and finalize the response
    - output guardrail: validate the output message"""

    input_guardrail: BaseGuardRail = Field(
        default=PassGuardRail(),
        description="validate the input message",
    )
    prompt_augmenter: BasePromptAugmenter = Field(
        default=IdentityPromptAugmenter(),
        description="add more context to the prompt or rewrite / refine the prompt",
    )
    tools: Sequence[Callable] = Field(
        default=[],
        description="tools to call for the LLM",
    )
    output_guardrail: BaseGuardRail = Field(
        default=PassGuardRail(),
        description="validate the output message",
    )

    @abc.abstractmethod
    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # call tool if any and finalize response
        pass

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.input_guardrail(message)
        message = self.prompt_augmenter(message)
        message = self.generate(message, **kwargs)
        message = self.output_guardrail(message)
        return message


class MetaPromptAugmenter(BasePromptAugmenter):
    """A prompt augmenter that use an LLM chain to rewrite the prompt."""

    chain: BaseLLMChain = Field(..., description="LLM chain that rewrite the prompt")

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.chain.invoke(message, **kwargs)
        return message
