from collections.abc import Sequence
from typing import Dict, List, Tuple

from adp_core import utils
from adp_core.agent import AgentMessage
from adp_core.chain import BaseCausalMultiTurnsChain
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from pydantic import Field, PrivateAttr


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[ChatMessage, ChatResponse]):
    model: FunctionCallingLLM = Field(
        ..., description="The LLM model with function calling capability"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")

    _tool_dict: Dict[str, BaseTool] = PrivateAttr({})

    def __init__(self, tools: Sequence[BaseTool] = [], **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def _prepare_conversation(self, message: AgentMessage) -> List[ChatMessage]:
        conversation = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            ChatMessage(
                role=MessageRole.USER,
                content=self.user_prompt_template.format(**message.to_dict()),
            ),
        ]
        for prev_response in message.responses:
            if prev_response[0] == "user":
                conversation.append(
                    ChatMessage(role=MessageRole.USER, content=prev_response[1])
                )
            elif prev_response[0] == "tool":
                conversation.append(
                    ChatMessage(role=MessageRole.TOOL, content=prev_response[1])
                )
            elif prev_response[0] == "system":
                conversation.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=prev_response[1])
                )
            else:
                conversation.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=prev_response[1])
                )
        return conversation

    def _generate_response(
        self, message: AgentMessage, conversation: List[ChatMessage], **kwargs
    ) -> Tuple[AgentMessage, ChatResponse, bool]:
        response = self.model.chat(conversation, tools=self.tools, **kwargs)
        response.message.content = utils.remove_thinking(str(response.message.content))
        message.responses.append((self.name, str(response.message.content)))
        has_tool = bool(
            self.model.get_tool_calls_from_response(
                response, error_on_no_tool_call=False
            )
        )
        return message, response, has_tool

    def _process_tools(
        self,
        message: AgentMessage,
        conversation: List[ChatMessage],
        response: ChatResponse,
    ) -> Tuple[AgentMessage, List[ChatMessage]]:
        conversation.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=utils.remove_thinking(str(response.message.content)),
            )
        )
        tool_calls = self.model.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        # Source: https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/#the-workflow-itself
        # call tools -- safely!
        for tool_call in tool_calls:
            tool = self._tool_dict.get(tool_call.tool_name)
            if tool is None:
                res = f"Tool {tool_call.tool_name} does not exist"
                conversation.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=res,
                        additional_kwargs={},
                    )
                )
                message.responses.append((MessageRole.TOOL, res))
            else:
                additional_kwargs = {
                    "tool_call_id": tool_call.tool_id,
                    "name": tool.metadata.get_name(),
                }
                try:
                    tool_output = tool(**tool_call.tool_kwargs)
                    conversation.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=tool_output.content,
                            additional_kwargs=additional_kwargs,
                        )
                    )
                    message.responses.append((MessageRole.TOOL, tool_output.content))
                except Exception as e:
                    res = f"Encountered error in tool call: {tool_call.tool_name} {e}"
                    conversation.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=res,
                            additional_kwargs=additional_kwargs,
                        )
                    )
                    message.responses.append((MessageRole.TOOL, res))

        return message, conversation

    @property
    def tools(self) -> List[BaseTool]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[BaseTool]):
        self._tool_dict = {tool.metadata.get_name(): tool for tool in tools}
