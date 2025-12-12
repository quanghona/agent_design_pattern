import abc
from typing import Any, Dict, Generic, List, Tuple, TypeVar
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, AgentResponse
import dspy
from pydantic import Field, PrivateAttr


Signature = TypeVar("Signature", bound=dspy.Signature)


class BaseSignatureAdapter(abc.ABC, Generic[Signature]):
    """The adapter convert between AgentMessage and dspy.Signature
    In this class we also have the prefill dictionary to fill in values to the Signature fields.
    This is useful when we have static fields that don't exist in the AgentMessage object while it is moving in the workflow"""

    _prefill_dict: Dict[str, Any] = PrivateAttr({})

    @abc.abstractmethod
    def msg2sig(self, message: AgentMessage) -> List[Signature]:
        """The signature fields are only known when developing end application.
        This function convert AgentMessage fields to dspy Signature before flow into the dspy predictor.
        The filling logic for signature should be implemented in this method in the child class

        Specifically, there are 2 attributes need to taken care of:
        - prefill dictionary in this adapter class. This is also known as the static filling
        - the context dictionary in the AgentMessage. This is also known as the dynamic filling

        Args:
            message (AgentMessage): message to convert

        Returns:
            List[Signature]: list of the conversation so fat in dspy.Signature format"""
        raise NotImplementedError

    @abc.abstractmethod
    def sig2msg(self, signatures: List[Signature], name: str) -> List[AgentResponse]:
        """dspy.Signature to AgentMessage mapping.
          This function used after the flow is completed and the dspy output need to convert back to the agent message.

          Note about extracting the source name who generate the message and the message content from signature.
          The source can be assistant or tool, the user message is the dspy.InputField, and dspy already handled the system message.
          To unify about the source name, we can make the following assumptions:
          If a signature have both OutputField and ToolCalls, it is a tool message. Otherwise it is an assistant message

        Args:
            signatures (List[Signature]): dspy output
            name (str): agent name

        Returns:
            List[AgentResponse]: list of responses extract from the signatures input
        """
        raise NotImplementedError

    @classmethod
    def with_prefill(cls, prefill_dict: Dict[str, str]) -> "BaseSignatureAdapter":
        """Create a new instance of the adapter with the given prefill dictionary.
        Note that the child class is responsible for manage the matching and evaluation between prefill dictionary and signature

        Args:
            prefill_dict (Dict[str, str]): The prefill dictionary.

        Returns:
            BaseSignatureAdapter: A new instance of the adapter with the given prefill dictionary.
        """
        obj = cls()
        obj._prefill_dict = prefill_dict
        return obj

    def add_prefill(self, key: str, value: Any) -> None:
        """Add a new key-value pair to the prefill dictionary.

        Args:
            key (str): The key to add.
            value (str): The value to add.
        """
        self._prefill_dict[key] = value

    def remove_prefill(self, key: str) -> None:
        """Remove a key from the prefill dictionary.

        Args:
            key (str): The key to remove.
        """
        del self._prefill_dict[key]


class ChatCausalMultiTurnsChain(
    BaseCausalMultiTurnsChain[dspy.Signature, dspy.Prediction],
    arbitrary_types_allowed=True,
):
    """A class that handle LM call using dspy without history.

    Regarding tool calling pattern used in dspy framework, there are 2 approaches proposed by authors of dspy:
    1. [dspy fully managed](https://dspy.ai/learn/programming/tools/#approach-1-using-dspyreact-fully-managed): using dspy.ReAct or its subclass or customized dspy.Module that handle tool calling internally.
    In this case, first the signature of module doesn't have dspy.ToolCalls field. All tools completely stay inside the dspy module.
    This class only get the final output produced by dspy predictor. The _process_tools will not be call at all.

    2. [Manual tool handling](https://dspy.ai/learn/programming/tools/#approach-1-using-dspyreact-fully-managed): tool calling logic is handled by this class.
    When initializing this class with provided signature, this class will automaticallty detect the dspy.ToolCalls field.
    When invoke the chain, it will detects and calls the tool depends on the value of the tool calls field.

    Reference: https://dspy.ai
    """

    predictor: dspy.Module = Field(..., description="dspy predictor")
    adapter: BaseSignatureAdapter = Field(
        ...,
        description="The adapter convert between AgentMessage and dspy.Signature",
    )
    _signature: type[dspy.Signature] = PrivateAttr()
    _tool_calls_field: str | None = PrivateAttr(None)
    _lm: dspy.LM | None = PrivateAttr(None)
    _history_field_name: str | None = PrivateAttr(None)

    def __init__(self, signature: str | type[dspy.Signature], **kwargs):
        super().__init__(**kwargs)
        self._signature = dspy.ensure_signature(signature)
        for key, value in self._signature.input_fields.items():
            if value.annotation is dspy.History:
                self._history_field_name = key
                break
        for key, value in self._signature.output_fields.items():
            if value.annotation is dspy.ToolCalls:
                self._tool_calls_field = key
                break

    def _prepare_conversation(self, message: AgentMessage) -> List[dspy.Signature]:
        return self.adapter.msg2sig(message)

    def _generate_response(
        self, conversation: List[dspy.Signature], **kwargs
    ) -> Tuple[List[dspy.Signature], dspy.Prediction, bool]:
        sig = conversation[-1].model_dump(exclude_none=True)
        if self._history_field_name is not None:
            # Convert dict to object as the library use object to access the history
            history = dspy.History(messages=sig[self._history_field_name]["messages"])
            sig[self._history_field_name] = history
        if self._lm:
            # change context if possible
            with dspy.context(lm=self._lm):
                data = self.predictor(**sig)
        else:
            data = self.predictor(**sig)

        has_tool = (
            False
            if self._tool_calls_field is None
            else bool(data[self._tool_calls_field])
        )
        sig.update(data.items())
        conversation.append(self._signature(**sig))
        return conversation, data, has_tool

    def _process_tools(
        self, conversation: List[dspy.Signature], response: dspy.Prediction
    ) -> List[dspy.Signature]:
        for call in response[self._tool_calls_field].tool_calls:
            result = call.execute()
            for key, value in self._signature.output_fields.items():
                if value.annotation is str:
                    sig = self._signature(**response, **{key: result})
                    conversation.append(sig)
                    break

        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[dspy.Signature]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 1
            if self.store_immediate_steps
            else len(conversation) - 1
        )
        end_index = len(conversation)
        message.responses.extend(
            self.adapter.sig2msg(conversation[start_index:end_index], self.name)
        )
        # TODO: handle other modals later
        return message

    def with_lm(self, lm: dspy.LM | None) -> "ChatCausalMultiTurnsChain":
        """Set the language model context to use for the chain.
        If set to None, the default context will be used.

        Args:
            lm (dspy.LM | None): The language model context to use for the chain.

        Returns:
            ChatCausalMultiTurnsChain: The updated ChatCausalMultiTurnsChain object."""
        self._lm = lm
        return self
