import abc
from collections.abc import Sequence
from typing import Callable, Generic, List, Tuple

from pydantic import Field, PrivateAttr

from .guardrail import BaseGuardRail, PassGuardRail
from .prompt_enhancer import (
    BasePromptEnhancer,
    IdentityPromptEnhancer,
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
    _last_response_as_context: str | None = PrivateAttr(default=None)

    def response_as_context(self, key: str):
        self._last_response_as_context = key.replace("context_", "")

    @abc.abstractmethod
    def _prepare_conversation(self, message: AgentMessage) -> List[ChainMessage]:
        pass

    @abc.abstractmethod
    def _generate_response(
        self, message: AgentMessage, conversation: List[ChainMessage], **kwargs
    ) -> Tuple[AgentMessage, ChainResponse, bool]:
        pass

    @abc.abstractmethod
    def _process_tools(
        self,
        message: AgentMessage,
        conversation: List[ChainMessage],
        response: ChainResponse,
    ) -> Tuple[AgentMessage, List[ChainMessage]]:
        pass

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        # Template method
        conversation = self._prepare_conversation(message)
        message, response, has_tool = self._generate_response(
            message, conversation, **kwargs
        )
        if has_tool:
            message, conversation = self._process_tools(message, conversation, response)
            message, response, _ = self._generate_response(
                message, conversation, **kwargs
            )

        if self._last_response_as_context is not None:
            if message.context is None:
                message.context = {}
            message.context[self._last_response_as_context] = str(
                message.responses[-1][1]
            )

        message.origin = self.name
        message.execution_result = "success"
        return message


class TypicalLLMChain(BaseLLMChain):
    """
    A LLM chain that consists of all components in a typical response generation.
    The chain  have 4 steps:
    - input guardrail: validate the input message
    - prompt enhancement: add more context to the prompt or rewrite / refine the prompt
    - generate: generate the response. This steps should also handle the tool calling procedure and finalize the response
    - output guardrail: validate the output message"""

    input_guardrail: BaseGuardRail = Field(
        default=PassGuardRail(),
        description="validate the input message",
    )
    prompt_enhancer: BasePromptEnhancer = Field(
        default=IdentityPromptEnhancer(),
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
        message = self.prompt_enhancer(message)
        message = self.generate(message, **kwargs)
        message = self.output_guardrail(message)
        return message


class LLMPromptEnhancer(BasePromptEnhancer):
    """A prompt enhancer that use an LLM chain to rewrite the prompt."""

    chain: BaseLLMChain = Field(..., description="LLM chain that rewrite the prompt")

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.chain.invoke(message, **kwargs)
        if message.execution_result != "success":
            return message
        _, message.query = message.responses[-1]
        message.responses = []
        return message
