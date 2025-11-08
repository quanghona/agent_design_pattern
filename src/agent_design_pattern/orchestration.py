from typing import Callable, List
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
                 task_response_key: str = "context_response",
                 state_change_callback: Callable[[str], None] = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.chain_task = chain_task
        self.chain_reflection = chain_reflection
        self.task_response_key = task_response_key

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self._set_state("running")
        message = self.chain_task.invoke(message, **kwargs)
        if message.execution_result != "success":
            return message

        self._set_state("reflecting")
        message.context = {}
        message.context[self.task_response_key.replace("context_", "")] = message.response
        result_message = self.chain_reflection.invoke(message, **kwargs)

        self._set_state("idle")
        result_message.origin = self.name
        return result_message


class LoopAgent(IAgent):
    """
    An agent that will loop until a certain condition is met.

    This agent takes two parameters: an agent to loop and a condition to stop the loop.
    The condition to stop the loop should be a function that takes an AgentMessage as an argument
    and returns a boolean indicating whether the loop should stop or not.

    Example:
        def is_stop(message: AgentMessage) -> bool:
            # stop when the message query is "stop"
            return message.query == "stop"

        agent = LoopAgent(
            agent=Agent1(),
            is_stop=is_stop
        )

        message = AgentMessage(query="hello")
        result = agent.execute(message)

    Attributes:
        agent: The agent to loop.
        is_stop: The condition to stop the loop.
    """

    def __init__(self,
                 agent: IAgent,
                 is_stop: Callable[[AgentMessage], bool],
                 state_change_callback: Callable[[str], None] = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.agent = agent
        self.is_stop = is_stop

    def execute(self,
                message: AgentMessage,
                result_strategy: str = 'last',    # last, last_n, all, custom
                result_strategy_param: Callable[[str, str, List[str]], List[str]] | int | None = None,
                **kwargs) -> AgentMessage:
        assert result_strategy in ['last', 'last_n', 'all', 'custom'], f"result_strategy must be one of ['last', 'last_n', 'all', 'custom'], received {result_strategy}"
        if result_strategy == 'last_n':
            if not isinstance(result_strategy_param, int):
                raise ValueError(f"result_strategy_param must be int when result_strategy is 'last_n', received {type(result_strategy_param)}")
            elif result_strategy_param <= 0:
                raise ValueError(f"result_strategy_param must be greater than 0 when result_strategy is 'last_n', received {result_strategy_param}")
        elif result_strategy == 'custom' and not isinstance(result_strategy_param, Callable[[int, str], bool]):
            raise ValueError("result_strategy_param must be a callable function when result_strategy is 'custom'")

        self._set_state("running")
        message.responses = []
        if result_strategy == 'last':
            result_strategy = 'last_n'
            result_strategy_param = 1

        while self.is_stop(message) is False:
            message = self.agent.execute(message, **kwargs)
            if message.execution_result != "success":
                break

            message.responses.append((self.agent.name, message.response))
            if result_strategy == 'last_n':
                message.responses = message.responses[-result_strategy_param:]
            elif result_strategy == 'custom':
                message.responses = result_strategy_param(self.agent.name, message, message.responses)

        self._set_state("idle")
        message.origin = self.name
        return message


class SequentialAgent(IAgent):
    """
    A sequential agent is an agent that will execute a list of agents in a particular order.

    The order of the agents is the order of the execution. This means that the first agent will be executed first, then the second agent and so on.

    Example:
        agent1 = Agent1()
        agent2 = Agent2()

        sequential_agent = SequentialAgent([agent1, agent2])

        message = AgentMessage(query="hello")
        result = sequential_agent.execute(message)

    Attributes:
        agents: The list of agents to execute in sequence.
    """
    def __init__(self, agents: List[IAgent], state_change_callback: Callable[[str], None] = None, name: str = None, **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.agents = agents

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        for agent in self.agents:
            self._set_state("running agent " + agent.name)
            message = agent.execute(message, **kwargs)

            if message.execution_result != "success":
                break

        self._set_state("idle")
        message.origin = self.name

        return message

