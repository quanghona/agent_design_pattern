from collections.abc import Callable, Sequence
from typing import Dict, List, Tuple

from adp_core import utils
from adp_core.agent import AgentMessage
from adp_core.chain import BaseCausalMultiTurnsChain
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class ChatCausalMultiTurnsChain(BaseCausalMultiTurnsChain[BaseMessage, AIMessage]):
    # For resuability at runtime, wen need to control the assignment and rebuild
    # the model with its partners to operate correctly
    _model: BaseChatModel = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    _user_prompt_template: str = PrivateAttr()
    _prompt: ChatPromptTemplate = PrivateAttr()
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
        self._prompt = ChatPromptTemplate(
            [("system", system_prompt), ("human", user_prompt_template)]
        )
        self.bind_tools(tools, tool_choice=tool_choice)

    def _prepare_conversation(self, message: AgentMessage) -> List[BaseMessage]:
        conversation = [
            SystemMessage(self._system_prompt),
            HumanMessage(self._user_prompt_template.format(**message.to_dict())),
        ]
        if message.responses is not None and len(message.responses) > 0:
            for previous_response in message.responses:
                if previous_response[0] == "user":
                    conversation.append(HumanMessage(previous_response[1]))
                elif previous_response[0] == "tool":
                    conversation.append(ToolMessage(previous_response[1]))
                elif (
                    previous_response[0] == "system"
                ):  # this shouldn't happen but just in case
                    conversation.append(SystemMessage(previous_response[1]))
                else:
                    conversation.append(AIMessage(previous_response[1]))
        return conversation

    def _generate_response(
        self, message: AgentMessage, conversation: List[BaseMessage], **kwargs
    ) -> Tuple[AgentMessage, AIMessage, bool]:
        response = self._model.invoke(conversation, **kwargs)
        response.content = utils.remove_thinking(str(response.content))
        message.responses.append((self.name, response.content))
        return message, response, len(response.tool_calls) > 0

    def _process_tools(
        self,
        message: AgentMessage,
        conversation: List[BaseMessage],
        response: AIMessage,
    ) -> Tuple[AgentMessage, List[BaseMessage]]:
        conversation.append(AIMessage(utils.remove_thinking(str(response.content))))
        for tool_call in response.tool_calls:
            if tool_call["name"] not in self._tool_dict:
                res = tool_call["name"] + " does not exist"
                conversation.append(
                    ToolMessage(
                        res,
                        tool_call_id=tool_call["id"],
                    )
                )
                message.responses.append(("tool", res))
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
                    message.responses.append(("tool", str(tool_response)))
                except Exception as e:
                    res = (
                        f"Encounter error while executing tool {tool_call['name']}. {e}"
                    )
                    conversation.append(ToolMessage(res, tool_call_id=tool_call["id"]))
                    message.responses.append(("tool", res))
        return message, conversation

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
            self._tool_dict = {tool.__name__: tool for tool in tools}
            self._chain = self._prompt | model_with_tools
        else:
            self._chain = self._prompt | self._model

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
        self._prompt = ChatPromptTemplate(
            [("system", system_prompt), ("human", user_prompt_template)]
        )
        if len(self._tool_dict) > 0:
            model_with_tools = self._model.bind_tools(
                list(self._tool_dict.values()), tool_choice=self._tool_choice
            )
            self._chain = self._prompt | model_with_tools
        else:
            self._chain = self._prompt | self._model

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
            self._chain = self._prompt | model_with_tools
        else:
            self._chain = self._prompt | self._model
