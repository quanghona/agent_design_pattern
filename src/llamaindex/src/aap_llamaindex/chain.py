from collections.abc import Callable, Sequence
from typing import Dict, List, Tuple

from aap_core import utils
from aap_core.agent import AgentMessage
from aap_core.chain import BaseCausalMultiTurnsChain
from llama_index.core.base.llms.types import MessageRole, TextBlock
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool
from pydantic import Field, PrivateAttr


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[ChatMessage, ChatResponse]):
    model: FunctionCallingLLM = Field(
        ..., description="The LLM model with function calling capability"
    )
    system_prompt: str = Field(..., description="The system prompt")
    user_prompt_template: str = Field(..., description="The user prompt template")

    _tool_dict: Dict[str, BaseTool] = PrivateAttr({})

    def __init__(self, tools: Sequence[BaseTool | Callable] = [], **kwargs):
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
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append(
                    ChatMessage(role=MessageRole.USER, content=response[1])
                )
            elif response[0] == "tool":
                conversation.append(
                    ChatMessage(role=MessageRole.TOOL, content=response[1])
                )
            elif response[0] == "system":
                conversation.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=response[1])
                )
            else:
                conversation.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=response[1])
                )
        return conversation

    def _generate_response(
        self, conversation: List[ChatMessage], **kwargs
    ) -> Tuple[List[ChatMessage], ChatResponse, bool]:
        response = self.model.chat_with_tools(
            user_msg=conversation[-1],
            chat_history=conversation[:-1],
            tools=self.tools,
            **kwargs,
        )
        has_tool = False
        for block in response.message.blocks:
            if block.block_type == "text" and len(block.text) > 0:
                block.text = utils.remove_thinking(block.text)
            elif block.block_type == "tool_call":
                has_tool = True
        conversation.append(response.message)
        return conversation, response, has_tool

    def _process_tools(
        self, conversation: List[ChatMessage], response: ChatResponse
    ) -> List[ChatMessage]:
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
                except Exception as e:
                    res = f"Encountered error in tool call: {tool_call.tool_name} {e}"
                    conversation.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=res,
                            additional_kwargs=additional_kwargs,
                        )
                    )

        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[ChatMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            MessageRole.ASSISTANT: self.name,
            MessageRole.CHATBOT: self.name,
            MessageRole.FUNCTION: "tool",
            MessageRole.DEVELOPER: "system",
            MessageRole.MODEL: self.name,
            MessageRole.USER: "user",
            MessageRole.TOOL: "tool",
            MessageRole.SYSTEM: "system",
        }
        for i in range(start_index, end_index):
            if len(conversation[i].blocks) == 1 and isinstance(
                conversation[i].blocks[0], TextBlock
            ):
                message.responses.append(
                    (name_map[conversation[i].role], conversation[i].blocks[0].text)  # type: ignore
                )
            # TODO: handle other modals later

        return message

    @property
    def tools(self) -> List[BaseTool]:
        return list(self._tool_dict.values())

    @tools.setter
    def tools(self, tools: Sequence[BaseTool | Callable]):
        if len(tools) > 0 and isinstance(tools[0], Callable):
            tools = [FunctionTool.from_defaults(tool) for tool in tools]
        self._tool_dict = {tool.metadata.get_name(): tool for tool in tools}  # type: ignore
