from collections.abc import Callable, Sequence
from typing import Dict, List, Tuple

from aap_core import utils
from aap_core.agent import AgentMessage
from aap_core.chain import BaseCausalMultiTurnsChain
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[BaseMessage, AIMessage]):
    # For resuability at runtime, wen need to control the assignment and rebuild
    # the model with its partners to operate correctly
    _model: BaseChatModel = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    _user_prompt_template: str = PrivateAttr()
    _tool_dict: Dict[str, Callable | BaseTool] = PrivateAttr({})
    _tool_choice: str | None = PrivateAttr()
    _chain = PrivateAttr()

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str,
        user_prompt_template: str = "{query}",
        tools: Sequence[Callable | BaseTool] = [],
        tool_choice: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self.bind_tools(tools, tool_choice=tool_choice)

    def _prepare_conversation(self, message: AgentMessage) -> List[BaseMessage]:
        conversation = [
            SystemMessage(self._system_prompt),
            HumanMessage(self._user_prompt_template.format(**message.to_dict())),
        ]
        total_turns = (
            min(len(message.responses), self.include_history)
            if self.include_history >= 0
            else len(message.responses)
        )
        responses = message.responses[-total_turns:]
        for response in responses:
            if response[0] == "user":
                conversation.append(HumanMessage(response[1]))
            elif response[0] == "tool":
                conversation.append(ToolMessage(response[1]))
            elif response[0] == "system":
                # this shouldn't happen but just in case
                conversation.append(SystemMessage(response[1]))
            else:
                conversation.append(AIMessage(response[1]))
        return conversation

    def _generate_response(
        self, conversation: List[BaseMessage], **kwargs
    ) -> Tuple[List[BaseMessage], AIMessage, bool]:
        response = self._chain.invoke(conversation, **kwargs)
        if len(response.content) > 0 and isinstance(response.content, str):
            response.content = utils.remove_thinking(str(response.content))
        conversation.append(response)
        return conversation, response, len(response.tool_calls) > 0

    def _process_tools(
        self,
        conversation: List[BaseMessage],
        response: AIMessage,
    ) -> List[BaseMessage]:
        for tool_call in response.tool_calls:
            if tool_call["name"] not in self._tool_dict:
                res = tool_call["name"] + " does not exist"
                conversation.append(ToolMessage(res, tool_call_id=tool_call["id"]))
            else:
                try:
                    tool_func = self._tool_dict[tool_call["name"]]
                    if isinstance(tool_func, BaseTool):
                        tool_response = tool_func.invoke(tool_call["args"])
                    elif isinstance(tool_func, Callable):
                        tool_response = tool_func(**tool_call["args"])
                    else:
                        raise ValueError(
                            f"Tool {tool_call['name']} is not a BaseTool or Callable"
                        )
                    conversation.append(
                        ToolMessage(str(tool_response), tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    res = (
                        f"Encounter error while executing tool {tool_call['name']}. {e}"
                    )
                    conversation.append(ToolMessage(res, tool_call_id=tool_call["id"]))
        return conversation

    def _append_responses(
        self, message: AgentMessage, conversation: List[BaseMessage]
    ) -> AgentMessage:
        start_index = (
            min(len(message.responses), self.include_history) + 2
            if self.store_immediate_steps
            else len(conversation) - 1
        )  # 2 is system message and user query
        end_index = len(conversation)
        name_map = {
            "ai": self.name,
            "human": "user",
            "tool": "tool",
            "system": "system",
        }
        for i in range(start_index, end_index):
            if (
                isinstance(conversation[i].content, str)
                and len(conversation[i].content) > 0
            ):
                message.responses.append(
                    (name_map[conversation[i].type], conversation[i].content)
                )
            # TODO: handle other modals later
        return message

    def bind_tools(
        self, tools: Sequence[Callable | BaseTool], tool_choice: str | None = None
    ) -> None:
        """
        Bind tools to the model.

        Args:
            tools (Sequence[Callable | BaseTool]): A sequence of tools to bind.
            tool_choice (str | None): Follow langchain tool_choice in the
                [bind_tools](https://reference.langchain.com/python/langchain_core/language_models/?h=bind_tools#langchain_core.language_models.BaseChatModel.bind_tools) method

        Returns:
            None
        """
        if len(tools) > 0:
            model_with_tools = self._model.bind_tools(tools, tool_choice=tool_choice)
            self._tool_choice = tool_choice
            if isinstance(tools[0], BaseTool):
                self._tool_dict = {tool.name: tool for tool in tools}  # type: ignore
            else:
                self._tool_dict = {tool.__name__: tool for tool in tools}
            self._chain = model_with_tools
        else:
            self._chain = self._model

    def update_prompt(self, system_prompt: str, user_prompt_template: str) -> None:
        """
        Update the system prompt and user prompt template of the model.
        this will rebuild the template and the LLM chain

        Args:
            system_prompt (str): The new system prompt.
            user_prompt_template (str): The new user prompt template.
        """
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        if len(self._tool_dict) > 0:
            model_with_tools = self._model.bind_tools(
                list(self._tool_dict.values()), tool_choice=self._tool_choice
            )
            self._chain = model_with_tools
        else:
            self._chain = self._model

    @property
    def model(self) -> BaseChatModel:
        return self._model

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_prompt_template(self) -> str:
        return self._user_prompt_template

    @property
    def tools(self) -> List[Callable | BaseTool]:
        return list(self._tool_dict.values())

    @model.setter
    def model(self, model: BaseChatModel):
        self._model = model
        if len(self._tool_dict) > 0:
            model_with_tools = self._model.bind_tools(
                list(self._tool_dict.values()),
                tool_choice=self._tool_choice,
            )
            self._chain = model_with_tools
        else:
            self._chain = self._model
