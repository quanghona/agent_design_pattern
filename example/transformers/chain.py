from typing import List
from agent_design_pattern import utils
from agent_design_pattern.agent import AgentMessage, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json


class CasualSingleTurnChain(LLMChain):
    def __init__(self, model, system_prompt, user_prompt_template = "{query}", device="cuda", tools = [], **kargs):
        super().__init__()
        self.device = device
        if isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            # drop device_map if running on CPU
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map=self.device)
            self.model.eval()
        else:
            self.tokenizer, self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.tools = tools
        self.tool_map = {tool.__name__: tool for tool in tools}

    def extract_json(input_str: str) -> List[dict]:
        pattern = r'<tool_call>(?s:.*?)<\/tool_call>'
        matches = re.findall(pattern, input_str)
        return [match.strip().replace("<tool_call>", "").replace("</tool_call>", "").replace("\\n", "") for match in matches]

    def parse_call_tools(self, input_str: str):
        tool_calls = self.extract_json(input_str)
        tool_results = []

        for tool_call in tool_calls:
            tool_json = json.loads(tool_call)
            if tool_json["name"] in self.tool_map.keys():
                tool_result = self.tool_map[tool_json["name"]](**tool_json["arguments"])
                tool_results.append(tool_result)

        return tool_calls, tool_results

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        user_prompt = self.user_prompt_template.format(**message.to_dict())
        chat_messages = [
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user_prompt },
        ]

        def get_response(chat):
            inputs = self.tokenizer.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            # generate output tokens
            output = self.model.generate(**inputs.to(self.device), **kwargs)
            # decode output tokens into text
            output = self.tokenizer.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            for i, out_i in enumerate(output):
                output[i] = utils.remove_thinking(out_i)
            return output

        output = get_response(chat_messages)
        if "<tool_call>" in output:
            try:
                tool_calls, tool_results = self.parse_call_tools(output)
                chat_messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call} for tool_call in tool_calls]})
                for tool_result in tool_results:
                    chat_messages.append({"role": "tool", "content": str(tool_result)})
            except Exception as e:
                chat_messages.append({"role": "assistant", "content": "Encountered error while calling tool\n" + str(e)})
            output = get_response(chat_messages)

        message.response = output[0]
        message.execution_result = "success"
        return message


class CasualMultiTurnsChain(LLMChain):
    def __init__(self, model, system_prompt, user_prompt_template = "{query}", device="cuda", tools = [], **kargs):
        super().__init__()
        self.device = device
        if isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            # drop device_map if running on CPU
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map=self.device)
            self.model.eval()
        else:
            self.tokenizer, self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.tools = tools
        self.tool_map = {tool.__name__: tool for tool in tools}

    def extract_json(input_str: str) -> List[dict]:
        pattern = r'<tool_call>(?s:.*?)<\/tool_call>'
        matches = re.findall(pattern, input_str)
        return [match.strip().replace("<tool_call>", "").replace("</tool_call>", "").replace("\\n", "") for match in matches]

    def parse_call_tools(self, input_str: str):
        tool_calls = self.extract_json(input_str)
        tool_results = []

        for tool_call in tool_calls:
            tool_json = json.loads(tool_call)
            if tool_json["name"] in self.tool_map.keys():
                tool_result = self.tool_map[tool_json["name"]](**tool_json["arguments"])
                tool_results.append(tool_result)

        return tool_calls, tool_results

    def invoke(self, message: AgentMessage, **kwargs) -> AgentMessage:
        user_prompt = self.user_prompt_template.format(**message.to_dict())
        chat_messages = [
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user_prompt },
        ]
        for message in message.responses:
            if message[0] == "user":
                chat_messages.append({"role": "user", "content": message[1]})
            elif message[0] == "tool":
                chat_messages.append({"role": "tool", "content": message[1]})
            else:
                chat_messages.append({"role": "assistant", "content": message[1]})

        def get_response(chat):
            inputs = self.tokenizer.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            output = self.model.generate(**inputs.to(self.device), **kwargs)
            return self.tokenizer.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        output = get_response(chat_messages)
        if "<tool_call>" in output:
            try:
                tool_calls, tool_results = self.parse_call_tools(output)
                chat_messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call} for tool_call in tool_calls]})
                for tool_result in tool_results:
                    chat_messages.append({"role": "tool", "content": str(tool_result)})
            except Exception as e:
                chat_messages.append({"role": "assistant", "content": "Encountered error while calling tool\n" + str(e)})
            output = get_response(chat_messages)

        message.response = output[0]
        message.execution_result = "success"
        return message
