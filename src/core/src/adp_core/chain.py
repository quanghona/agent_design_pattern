import abc
from typing import Callable, List

from .guardrail import BaseGuardRail, PassGuardRail
from .prompt_enhancer import (
    BasePromptEnhancer,
    IdentityPromptEnhancer,
)
from .types import AgentMessage, BaseChain


class BaseLLMChain(BaseChain):
    """
    Base class for LLM chains.
    """

    @abc.abstractmethod
    def generate(self, message: AgentMessage, **kwargs) -> AgentMessage:
        pass

    @abc.abstractmethod
    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.generate(message, **kwargs)

    async def ainvoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self.invoke(message, **kwargs)


class TypicalLLMChain(BaseLLMChain):
    """
    A LLM chain that consists of all components in a typical response generation.
    The chain  have 4 steps:
    - input guardrail: validate the input message
    - prompt enhancement: add more context to the prompt or rewrite / refine the prompt
    - generate: generate the response. This steps should also handle the tool calling procedure and finalize the response
    - output guardrail: validate the output message"""

    def __init__(
        self,
        input_guardrail: BaseGuardRail = PassGuardRail(),
        prompt_enhancer: BasePromptEnhancer = IdentityPromptEnhancer(),
        tools: List[Callable] = [],
        output_guardrail: BaseGuardRail = PassGuardRail(),
    ):
        super().__init__()
        self.input_guardrail = input_guardrail
        self.prompt_enhancer = prompt_enhancer
        self.tools = tools
        self.output_guardrail = output_guardrail

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

    def __init__(self, chain: BaseLLMChain, **kwargs):
        super().__init__()
        self.chain = chain

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        message = self.chain.invoke(message, **kwargs)
        message.query = str(message.response)
        message.response = None
        return message
