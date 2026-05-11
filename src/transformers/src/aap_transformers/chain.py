import json
import os
import re
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Literal, Tuple

from aap_core import utils
from aap_core.chain import BaseCausalMultiTurnsChain
from aap_core.types import AgentMessage, TokenUsage
from pydantic import Field, PrivateAttr
from typing_extensions import TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersChainMessage(TypedDict):
    role: str
    content: str


class ChatCausalMultiTurnsChain(
    BaseCausalMultiTurnsChain[TransformersChainMessage, str]
):
    _model: Any
    _tokenizer: Any = PrivateAttr()  # AutoTokenizer has type of Unknown!
    device: Literal["cpu", "cuda"] = Field(
        ..., description="Device the model to run on"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")
    _tool_dict: Dict[str, Callable] = PrivateAttr({})

    def __init__(
        self,
        model: str | os.PathLike | Tuple[Any, Any],
        tools: Sequence[Callable] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(model, str) or isinstance(model, os.PathLike):
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            # drop device_map if running on CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                model, device_map=self.device
            )
            self._model.eval()
        else:
            self._tokenizer, self._model = model
        self.tools = tools

    @classmethod
    def extract_json(cls, input_str: str) -> List[str]:
        pattern = r"<tool_call>(?s:.*?)<\/tool_call>"
        matches = re.findall(pattern, input_str)
        return [
            match.strip()
            .replace("<tool_call>", "")
            .replace("</tool_call>", "")
            .replace("\\n", "")
            for match in matches
        ]

    def _prepare_conversation(
        self, message: AgentMessage
    ) -> List[TransformersChainMessage]:
        user_prompt = self.user_prompt_template.format(**message.to_dict())
        conversation: List[TransformersChainMessage] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append({"role": "user", "content": response[1]})
            elif response[0] == "tool":
                conversation.append({"role": "tool", "content": response[1]})
            elif response[0] == "system":  # this shouldn't happen but just in case
                conversation.append({"role": "system", "content": response[1]})
            else:
                conversation.append({"role": "assistant", "content": response[1]})
        return conversation

    def _generate_response(
        self,
        conversation: List[TransformersChainMessage],
        **kwargs,
    ) -> Tuple[List[TransformersChainMessage], str, bool, TokenUsage]:
        inputs = self._tokenizer.apply_chat_template(
            conversation,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = self._model.generate(**inputs.to(self.device), **kwargs)
        input_tokens = inputs.input_ids.shape[-1]
        output_tokens = output[:, inputs.input_ids.shape[-1] :].shape[-1]
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        output = self._tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )[0]
        output = utils.remove_thinking(output)
        # TODO: aware with multimodal output
        conversation.append({"role": "assistant", "content": output})
        return conversation, output, "<tool_call>" in output, usage

    def _process_tools(
        self,
        conversation: List[TransformersChainMessage],
        response: str,
    ) -> List[TransformersChainMessage]:
        tool_calls = ChatCausalMultiTurnsChain.extract_json(response)

        for tool_call in tool_calls:
            tool_json = json.loads(tool_call)
            if tool_json["name"] not in self._tool_dict:
                res = f"Tool {tool_json['name']} does not exist"
                conversation.append({"role": "tool", "content": res})
            else:
                try:
                    tool_result = self._tool_dict[tool_json["name"]](
                        **tool_json["arguments"]
                    )
                    conversation.append({"role": "tool", "content": str(tool_result)})
                except Exception as e:
                    res = (
                        f"Encountered error while calling tool {tool_json['name']}. {e}"
                    )
                    conversation.append({"role": "tool", "content": res})
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[TransformersChainMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            "assistant": self.name,
            "user": "user",
            "tool": "tool",
            "system": "system",
        }
        for i in range(start_index, end_index):
            if (
                isinstance(conversation[i]["content"], str)
                and len(conversation[i]["content"]) > 0
            ):
                message.responses.append(
                    (name_map[conversation[i]["role"]], conversation[i]["content"])
                )
        # TODO: handle other modals later
        return message

    @property
    def tools(self) -> List[Callable]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[Callable]):
        self._tool_dict = {tool.__name__: tool for tool in tools}

    @property
    def model(self) -> Tuple[Any, Any]:
        return self._tokenizer, self._model

    @model.setter
    def model(self, model: str | os.PathLike | Tuple[Any, Any]):
        if isinstance(model, str) or isinstance(model, os.PathLike):
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            # drop device_map if running on CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                model, device_map=self.device
            )
            self._model.eval()
        else:
            self._tokenizer, self._model = model
