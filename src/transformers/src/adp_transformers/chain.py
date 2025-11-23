from collections.abc import Sequence
import json
import os
import re
from typing import Any, Callable, Dict, List, Literal, Tuple, TypedDict

from adp_core import utils
from adp_core.agent import AgentMessage
from adp_core.chain import BaseCausalMultiTurnsChain
from pydantic import Field, PrivateAttr
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersChainMessage(TypedDict):
    role: str
    content: str


class ChatCausalMultiTurnsChain(
    BaseCausalMultiTurnsChain[TransformersChainMessage, str]
):
    _model: _BaseModelWithGenerate
    _tokenizer: Any = PrivateAttr()  # AutoTokenizer has type of Unknown!
    device: Literal["cpu", "cuda"] = Field(
        ..., description="Device the model to run on"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")
    _tool_dict: Dict[str, Callable] = PrivateAttr({})

    def __init__(
        self,
        model: str | os.PathLike | Tuple[Any, _BaseModelWithGenerate],
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
        if message.responses is None:
            message.responses = []
        for msg in message.responses:
            if msg[0] == "user":
                conversation.append({"role": "user", "content": msg[1]})
            elif msg[0] == "tool":
                conversation.append({"role": "tool", "content": msg[1]})
            elif msg[0] == "system":  # this shouldn't happen but just in case
                conversation.append({"role": "system", "content": msg[1]})
            else:
                conversation.append({"role": "assistant", "content": msg[1]})
        return conversation

    def _generate_response(
        self,
        message: AgentMessage,
        conversation: List[TransformersChainMessage],
        **kwargs,
    ) -> Tuple[AgentMessage, str, bool]:
        inputs = self._tokenizer.apply_chat_template(
            conversation,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = self._model.generate(**inputs.to(self.device), **kwargs)
        output = self._tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )[0]
        output = utils.remove_thinking(output)
        message.responses.append((self.name, output))
        return message, output, "<tool_call>" in output

    def _process_tools(
        self,
        message: AgentMessage,
        conversation: List[TransformersChainMessage],
        response: str,
    ) -> Tuple[AgentMessage, List[TransformersChainMessage]]:
        conversation.append({"role": "assistant", "content": response})
        tool_calls = ChatCausalMultiTurnsChain.extract_json(response)

        for tool_call in tool_calls:
            tool_json = json.loads(tool_call)
            if tool_json["name"] not in self._tool_dict:
                res = f"Tool {tool_json['name']} does not exist"
                conversation.append({"role": "tool", "content": res})
                message.responses.append(("tool", res))
            else:
                try:
                    tool_result = self._tool_dict[tool_json["name"]](
                        **tool_json["arguments"]
                    )
                    conversation.append({"role": "tool", "content": str(tool_result)})
                    message.responses.append(("tool", str(tool_result)))
                except Exception as e:
                    res = (
                        f"Encountered error while calling tool {tool_json['name']}. {e}"
                    )
                    conversation.append({"role": "tool", "content": res})
                    message.responses.append(("tool", res))
        return message, conversation

    @property
    def tools(self) -> List[Callable]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[Callable]):
        self._tool_dict = {tool.__name__: tool for tool in tools}

    @property
    def model(self) -> Tuple[Any, _BaseModelWithGenerate]:
        return self._tokenizer, self._model

    @model.setter
    def model(self, model: str | os.PathLike | Tuple[Any, _BaseModelWithGenerate]):
        if isinstance(model, str) or isinstance(model, os.PathLike):
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            # drop device_map if running on CPU
            self._model = AutoModelForCausalLM.from_pretrained(
                model, device_map=self.device
            )
            self._model.eval()
        else:
            self._tokenizer, self._model = model
