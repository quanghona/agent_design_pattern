from typing import List
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.core.tools.types import BaseTool
from agent_design_pattern.agent import AgentMessage, LLMChain
from agent_design_pattern import utils


class CasualOllamaSingleTurnChain(LLMChain):
    def __init__(self, model, system_prompt, user_prompt_template, tools: List[BaseTool] = [], **kwargs):
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.tools = tools
        self.tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        if isinstance(model, str):
            self.llm = Ollama(model=model, **kwargs)
        else:
            self.llm = model

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=self.user_prompt_template.format(**message.to_dict())),
        ]
        response = self.llm.chat(messages, tools=self.tools, **kwargs)
        tool_calls = self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        if tool_calls:
            # Source: https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/#the-workflow-itself
            # call tools -- safely!
            for tool_call in tool_calls:
                tool = self.tools_by_name.get(tool_call.tool_name)
                additional_kwargs = {
                    "tool_call_id": tool_call.tool_id,
                    "name": tool.metadata.get_name(),
                }
                if not tool:
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=f"Tool {tool_call.tool_name} does not exist",
                            additional_kwargs=additional_kwargs,
                        )
                    )
                    continue

                try:
                    tool_output = tool(**tool_call.tool_kwargs)
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=tool_output.content,
                            additional_kwargs=additional_kwargs,
                        )
                    )
                except Exception as e:
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=f"Encountered error in tool call: {e}",
                            additional_kwargs=additional_kwargs,
                        )
                    )
            response = self.llm.chat(messages, tools=self.tools, **kwargs)

        response = utils.remove_thinking(response.message.content)

        message.response = response
        message.execution_result = "success"
        return message
