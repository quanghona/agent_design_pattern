from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
import re

from agent_design_pattern.agent import AgentMessage, LLMChain


class CasualOllamaSingleTurnChain(LLMChain):
    def __init__(self, model, system_prompt, user_prompt_template, tools=[], **kwargs):
        super().__init__()
        self.llm = ChatOllama(model=model, **kwargs) if isinstance(model, str) else model
        self.system_prompt = system_prompt
        self.user_prompt_template = PromptTemplate(template=user_prompt_template, input_variables=["query"])
        prompt = ChatPromptTemplate([
            ("system", system_prompt),
            ("human", user_prompt_template)
        ])
        if len(tools) > 0:
            self.llm.bind_tools(tools)
            self.tools = {tool.__name__: tool for tool in tools}
        self.chain = prompt | self.llm

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        response = self.chain.invoke(message.to_dict(), **kwargs)
        if response.tool_calls:
            prompt = [
                SystemMessage(self.system_prompt),
                HumanMessage(self.user_prompt_template.format(**message.to_dict())),
                response
            ]
            for tool_call in response.tool_calls:
                tool_response = self.tools[tool_call["name"]].invoke(tool_call["args"])
                prompt.append(ToolMessage(str(tool_response), tool_call_id=tool_call["id"]))
            response = self.llm.invoke(prompt, **kwargs)
            response = response.content
        else:
            response = response.content
        # Snippet to remove thinking
        if "</think>" in response and "<think>" not in response:
            response = "<think>" + response
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        message.response = response
        message.execution_result = "success"

        return message


class CasualOllamaMultiTurnsChain(LLMChain):
    def __init__(self, model, system_prompt, user_prompt_template, tools=[], **kwargs):
        super().__init__()
        self.llm = ChatOllama(model=model, **kwargs) if isinstance(model, str) else model
        self.system_prompt = system_prompt
        self.user_prompt_template = PromptTemplate(template=user_prompt_template, input_variables=["query"])
        if len(tools) > 0:
            self.llm.bind_tools(tools)
            self.tools = {tool.__name__: tool for tool in tools}

    def invoke(self, message, **kwargs) -> AgentMessage:
        prompt = [
            SystemMessage(self.system_prompt),
            HumanMessage(self.user_prompt_template.format(**message.to_dict())),
        ]
        if message.responses is not None and len(message.responses) > 0:
            for previous_response in message.responses:
                if previous_response[0] == "user":
                    prompt.append(HumanMessage(previous_response[1]))
                elif previous_response[0] == "tool":
                    prompt.append(ToolMessage(previous_response[1]))
                else:
                    prompt.append(AIMessage(previous_response[1]))
        response = self.llm.invoke(prompt, **kwargs)

        # Process tools
        if response.tool_calls:
            prompt.append(response)
            for tool_call in response.tool_calls:
                tool_response = self.tools[tool_call["name"]].invoke(tool_call["args"])
                prompt.append(ToolMessage(str(tool_response), tool_call_id=tool_call["id"]))
            response = self.llm.invoke(prompt, **kwargs)
        response = response.content

        # Snippet to remove thinking
        if "</think>" in response and "<think>" not in response:
            response = "<think>" + response
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if message.responses is None:
            message.responses = []
        message.responses.append((self.name, response))
        message.execution_result = "success"

        return message
