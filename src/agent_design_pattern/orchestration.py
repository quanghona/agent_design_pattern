import random
from typing import Callable, Generator, List, Literal, Tuple

import concurrent
from agent_design_pattern.agent import BaseAgent, AgentMessage, LLMChain


class ReflectionAgent(BaseAgent):
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
            message.error = "Call chain not success. Received: " + message.execution_result
            return message

        self._set_state("reflecting")
        message.context = {}
        message.context[self.task_response_key.replace("context_", "")] = message.response
        result_message = self.chain_reflection.invoke(message, **kwargs)

        self._set_state("idle")
        result_message.origin = self.name
        return result_message


class LoopAgent(BaseAgent):
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
                 agent: BaseAgent,
                 is_stop: Callable[[AgentMessage], bool] | Generator[bool, None, None],
                 state_change_callback: Callable[[str], None] = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.agent = agent
        self.is_stop = is_stop

    def execute(self,
                message: AgentMessage,
                keep_result: int | Callable[[str, str, List[str]], List[str]] = 1,
                **kwargs) -> AgentMessage:
        self._set_state("running")
        message.responses = []

        def update_responses():
            message.responses.append((self.agent.name, message.response))
            if isinstance(keep_result, int) and keep_result > 0:
                message.responses = message.responses[-keep_result:]
            elif isinstance(keep_result, Callable):
                message.responses = keep_result(self.agent.name, message, message.responses)

        i = 0
        if isinstance(self.is_stop, Callable):
            while self.is_stop(message) is False:
                self._set_state("running iteration " + str(i))
                i += 1
                message = self.agent.execute(message, **kwargs)
                if message.execution_result != "success":
                    break
                update_responses()
        elif isinstance(self.is_stop, Generator):
            try:
                while next(self.is_stop) is False:
                    self._set_state("running iteration " + str(i))
                    i += 1
                    message = self.agent.execute(message, **kwargs)
                    if message.execution_result != "success":
                        break
                    update_responses()
            except StopIteration as e:
                pass
        else:
            raise ValueError(f"is_stop must be a callable or a generator, received {type(self.is_stop)}")

        self._set_state("idle")
        message.origin = self.name
        return message


class SequentialAgent(BaseAgent):
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
    def __init__(self, agents: List[BaseAgent], state_change_callback: Callable[[str], None] = None, name: str = None, **kwargs):
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


class ParallelAgent(BaseAgent):
    """
    A parallel agent is an agent that will execute a list of agents in parallel.
    The agent will return a list of AgentMessage where each message is the result of the corresponding agent in the list.

    Example:
        agent1 = Agent1()
        agent2 = Agent2()

        parallel_agent = ParallelAgent([agent1, agent2])

        messages = [AgentMessage(query="hello"), AgentMessage(query="world")]
        results = parallel_agent.execute(messages)

    Attributes:
        agents: The list of agents to execute in parallel.
    """
    def __init__(self, agents: List[BaseAgent], state_change_callback: Callable[[str], None] = None, name: str = None, **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.agents = agents

    def execute(self, messages: AgentMessage | List[AgentMessage], **kwargs) -> AgentMessage:
        # If received a single message, all agents will process the same message
        if isinstance(messages, AgentMessage):
            messages = [messages] * len(self.agents)
        elif len(messages) != len(self.agents):
            raise ValueError("messages must be a list of AgentMessage with the same length as the number of agents")

        # if torch is used, we can use torch.multiprocessing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(agent.execute, message, **kwargs) for agent, message in zip(self.agents, messages)]
            messages = [future.result() for future in concurrent.futures.as_completed(futures)]

        result_message = AgentMessage(
            query="",
            origin=self.name,
            responses=[(agent.name, message.response) for agent, message in zip(self.agents, messages)],
            execution_result="success")
        return result_message


class CoordinatorAgent(BaseAgent):
    """
    This coordinator agent compose of 3 stages:
    - Planning: a planner agent will plan the steps to be executed.
    The planner recieves input query and returns a list of steps to be executed.
    For each step, it contains 3 fields:
        + worker: the worker agent that will execute the step
        + message: the instruction for the worker agent
        + dependencies: the dependencies of the step, this is picked from the result of the previous steps. The planner is responsible for choosing which results are needed for current step
    - Execution: For each step, an assigned worker will execute the step. In current version, steps are performed sequentially.
    - Summary: a summary agent will summarize the results so far and generate final answer.
    The summary agent recieves all results and write a final answer. This stage is optional.

    Example:
        planner_agent = PlannerAgent()
        worker_agent = WorkerAgent()
        summary_agent = SummaryAgent()

        coordinator_agent = CoordinatorAgent(planner_agent, worker_agent, summary_agent)

        message = AgentMessage(query="hello")
        result = coordinator_agent.execute(message)

    Attributes:
        planner_agent: The agent that will plan the steps to be executed.
        workers: The list of agents that will execute the steps.
        summary_agent: The agent that will summarize the results so far and generate final answer.
    """
    def __init__(self,
                 planner_agent: BaseAgent,
                 parse_plan: Callable[[str, List[BaseAgent]], List[Tuple[AgentMessage, BaseAgent]]],
                 workers: List[BaseAgent],
                 summary_chain: LLMChain = None,
                 summary_steps_key: str = "context_results",
                 state_change_callback: Callable[[str], None] | None = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.planner_agent = planner_agent
        self.parse_plan = parse_plan
        self.workers = {worker.__name__: worker for worker in workers}
        self.summary_chain = summary_chain
        if not summary_steps_key.startswith("context_"):
            raise ValueError("summary_steps_key must start with 'context_'")
        self.summary_steps_key = summary_steps_key.replace("context_", "")

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self._set_state("planning")
        message = self.coordinator.execute(message, **kwargs)

        if message.execution_result != "success":
            return message

        steps = self.parse_plan(message.response, self.workers.values())
        sub_task_result = []
        # TODO: check step dependency and parallelize steps to speed up
        for i, step_msg, step_agent, dependencies in enumerate(steps):
            step_msg.context = {self.summary_steps_key: sub_task_result}
            self._set_state(f"step {i}: worker {step_agent.name} running")
            sub_task_result.append(step_agent.execute(step_msg, **kwargs))

        if self.summary_chain is None:
            if self.summary_prompt is not None:
                # Use coordinator as summary agent
                self._set_state(f"{self.coordinator.name} finalizing")
                message.query = self.summary_prompt
                message.context = {self.summary_steps_key: sub_task_result}
                message = self.coordinator.execute(message, **kwargs)
            # if no summary prompt is provided, just return the result from workers
        else:
            self._set_state("finalizing")
            message.context = {self.summary_steps_key: sub_task_result}
            message = self.summary_chain.execute(message, **kwargs)

        self._set_state("idle")
        message.origin = self.name
        return message


class DebateAgent(BaseAgent):
    def __init__(self,
                 agents: List[BaseAgent],
                 pick_strategy: Literal["round_robin", "random"] | Callable[[List[BaseAgent]], BaseAgent] = "round_robin",
                 max_turns: int = 5,
                 should_stop: Callable[[AgentMessage], bool] = None,
                 state_change_callback: Callable[[str], None] = None,
                 name: str = None,
                 **kwargs):
        super().__init__(state_change_callback=state_change_callback, name=name, **kwargs)
        self.agents = agents
        self.pick_strategy = pick_strategy
        self.max_turns = max_turns
        self.should_stop = should_stop

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        n = 0
        message.responses = []

        # All response turn are stored in the message.responses
        while n < self.max_turns:
            if self.pick_strategy == "round_robin":
                next_turn_agent = self.agents[n % len(self.agents)]
            elif self.pick_strategy == "random":
                next_turn_agent = random.choice(self.agents)
            elif isinstance(self.pick_strategy, Callable):
                next_turn_agent = self.pick_strategy(self.agents)
            else:
                raise ValueError("pick_strategy must be one of 'round_robin', 'random' or a function")

            self._set_state(f"turn {n}: agent {next_turn_agent.name} running")
            message = next_turn_agent.execute(message, **kwargs)
            message.responses.append((next_turn_agent.name, message.response))
            n += 1

            if message.execution_result != "success" or self.should_stop(message):
                break

        self._set_state("idle")
        message.origin = self.name

        return message
