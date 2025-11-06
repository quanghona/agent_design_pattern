from typing import Callable
from agent_design_pattern.agent import IAgent, AgentMessage, LLMChain


class ReflectionAgent(IAgent):
    """
    An agent that is capable of self-reflection.

    This agent will first execute the given message and then reflect on the
    result. If the reflection is successful, the agent will return a new
    message with the reflection result. If the reflection is not successful,
    the agent will return the original message.

    Attributes:
        chain: The LLMChain to use for executing the message.
        chain_reflection: The prompt to use for self-reflection.
    """
    def __init__(self,
                 chain_task: LLMChain,
                 chain_reflection: LLMChain,
                 task_response_key: str = "artifact_response",
                 state_change_callback: Callable[[str], None] = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.chain_task = chain_task
        self.chain_reflection = chain_reflection
        self.task_response_key = task_response_key

    def execute(self, message: AgentMessage, reflection: AgentMessage, **kwargs) -> AgentMessage:
        self._set_state("running")
        message = self.chain_task.invoke(message, **kwargs)
        if message.execution_result != "success":
            return message

        self._set_state("reflecting")
        if reflection.artifact is None:
            reflection.artifact = {}
        reflection.artifact[self.task_response_key.replace("artifact_", "")] = message.response
        result_message = self.chain_reflection.invoke(reflection, **kwargs)

        self._set_state("idle")
        result_message.origin = self.name
        return result_message
