import concurrent.futures
import random
from typing import Callable, Dict, Generator, List, Literal, Tuple
from a2a.types import AgentCard

from pydantic import Field
from sacrebleu import sentence_bleu

from .agent import AgentMessage, BaseAgent
from .chain import BaseLLMChain


class ReflectionAgent(BaseAgent):
    """
    An agent that is capable of self-reflection.

    This agent will first execute the given message and then reflect on the
    result. If the reflection is successful, the agent will return a new
    message with the reflection result. If the reflection is not successful,
    the agent will return the original message.

    Attributes:
        chain: The BaseLLMChain to use for executing the message.
        chain_reflection: The prompt to use for self-reflection.
    """

    chain_task: BaseLLMChain = Field(
        ..., description="LLM chain that perform the main task"
    )
    chain_reflection: BaseLLMChain = Field(
        ...,
        description="LLM chain that perform reflection on the result of the main task",
    )
    task_response_key: str = Field(
        "context_response",
        description="""The key of which to store the context (e.g. intermediate result) during generation.
        Note that the key need to start with 'context_'""",
    )

    def __init__(
        self,
        card: AgentCard,
        chain_task: BaseLLMChain,
        chain_reflection: BaseLLMChain,
        task_response_key: str = "context_response",
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.chain_task = chain_task
        self.chain_reflection = chain_reflection
        self.task_response_key = task_response_key

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        message = self.chain_task.invoke(message, **kwargs)
        message.execution_result = message.execution_result or "error"
        if message.execution_result != "success":
            message.error_message = (
                "Call chain not success. Received: " + message.execution_result
            )
            return message

        self.state = "reflecting"
        message.context = {}
        message.context[self.task_response_key.replace("context_", "")] = (
            message.response
        )
        result_message = self.chain_reflection.invoke(message, **kwargs)

        self.state = "idle"
        result_message.origin = self.card.name
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

    agent: BaseAgent = Field(..., description="The agent to loop.")
    is_stop: Callable[[AgentMessage], bool] | Generator[bool, None, None] = Field(
        ...,
        description="The condition to stop the loop. It should be a function that takes an AgentMessage as an argument and returns a boolean.",
    )

    def __init__(
        self,
        card: AgentCard,
        agent: BaseAgent,
        is_stop: Callable[[AgentMessage], bool] | Generator[bool, None, None],
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.agent = agent
        self.is_stop = is_stop
        self.agent.state_change_callback = self._child_state_observer

    def execute(
        self,
        message: AgentMessage,
        keep_result: int
        | Callable[[str, str, List[Tuple[str, str]]], List[Tuple[str, str]]] = 1,
        **kwargs,
    ) -> AgentMessage:
        self.state = "running"
        message.responses = []

        def update_responses():
            if message.responses is None:
                message.responses = []
            message.responses.append((self.agent.card.name, message.response or ""))
            if isinstance(keep_result, int) and keep_result > 0:
                message.responses = message.responses[-keep_result:]
            elif isinstance(keep_result, Callable):
                message.responses = keep_result(
                    self.agent.card.name, str(message.response), message.responses
                )

        i = 0
        if isinstance(self.is_stop, Callable):
            while self.is_stop(message) is False:
                self.state = "running#" + str(i)
                i += 1
                message = self.agent.execute(message, **kwargs)
                if message.execution_result != "success":
                    break
                update_responses()
        elif isinstance(self.is_stop, Generator):
            try:
                while next(self.is_stop) is False:
                    self.state = "running#" + str(i)
                    i += 1
                    message = self.agent.execute(message, **kwargs)
                    if message.execution_result != "success":
                        break
                    update_responses()
            except StopIteration:
                pass
        else:
            raise ValueError(
                f"is_stop must be a callable or a generator, received {type(self.is_stop)}"
            )

        self.state = "idle"
        message.origin = self.card.name
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, [self.agent], "sequential"
        )


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

    agents: List[BaseAgent] = Field(
        ...,
        description="""The list of agents to execute in sequence.
            The order of the sequence is the order of the execution.""",
    )

    def __init__(
        self,
        card: AgentCard,
        agents: List[BaseAgent],
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.agents = agents
        for agent in self.agents:
            agent.state_change_callback = self._child_state_observer

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        for agent in self.agents:
            message = agent.execute(message, **kwargs)
            if message.execution_result != "success":
                break

        self.state = "idle"
        message.origin = self.card.name
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "sequential"
        )


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

    agents: List[BaseAgent] = Field(
        ..., description="""The list of agents to execute in parallel"""
    )

    def __init__(
        self,
        card: AgentCard,
        agents: List[BaseAgent],
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.agents = agents
        for agent in self.agents:
            agent.state_change_callback = self._child_state_observer

    def execute(  # type: ignore
        self, messages: AgentMessage | List[AgentMessage], **kwargs
    ) -> AgentMessage:
        # If received a single message, all agents will process the same message
        if isinstance(messages, AgentMessage):
            messages = [messages] * len(self.agents)
        elif len(messages) != len(self.agents):
            raise ValueError(
                "messages must be a list of AgentMessage with the same length as the number of agents"
            )

        self.state = "running"
        # if torch is used, we can use torch.multiprocessing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(agent.execute, message, **kwargs)
                for agent, message in zip(self.agents, messages)
            ]
            messages = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        result_message = AgentMessage(
            query="",
            origin=self.card.name,
            responses=[
                (agent.card.name, message.response)
                for agent, message in zip(self.agents, messages)
            ],
            execution_result="success",
        )  # type: ignore
        self.state = "idle"
        return result_message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "parallel"
        )


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
    The summary agent recieves all results and write a final answer.
    This stage is optional.

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

    planner_agent: BaseAgent = Field(
        ..., description="""The agent that will plan the steps to be executed"""
    )
    workers: Dict[str, BaseAgent] = Field(
        ...,
        description="""The list of agents that will execute the steps based on the plan""",
    )
    summary_chain: BaseLLMChain | None = Field(
        None,
        description="""The summary chain that will summarize the results so far and generate final answer""",
    )
    summary_prompt: str | None = Field(
        None, description="""The prompt that will be used by the summary chain"""
    )
    summary_steps_key: str = Field(
        "context_results",
        description="""The key of which to store the context (e.g. intermediate result) during generation.
        Note that the key need to start with 'context_'""",
    )

    def __init__(
        self,
        card: AgentCard,
        planner_agent: BaseAgent,
        parse_plan: Callable[
            [str, List[BaseAgent]], List[Tuple[AgentMessage, BaseAgent, List]]
        ],
        workers: List[BaseAgent],
        summary_chain: BaseLLMChain | None = None,
        summary_prompt: str | None = None,
        summary_steps_key: str = "context_results",
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.planner_agent = planner_agent
        self.parse_plan = parse_plan
        self.workers = {worker.__name__: worker for worker in workers}
        self.summary_chain = summary_chain
        self.summary_prompt = summary_prompt
        if not summary_steps_key.startswith("context_"):
            raise ValueError("summary_steps_key must start with 'context_'")
        self.summary_steps_key = summary_steps_key.replace("context_", "")

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "planning"
        message = self.planner_agent.execute(message, **kwargs)

        if message.execution_result != "success":
            return message

        steps = self.parse_plan(str(message.response), list(self.workers.values()))
        sub_task_result = []
        # TODO: check step dependency and parallelize steps to speed up
        for i, (step_msg, step_agent, dependencies) in enumerate(steps):
            step_msg.context = {self.summary_steps_key: sub_task_result}
            self.state = f"step {i}: worker {step_agent.card.name} running"
            sub_task_result.append(step_agent.execute(step_msg, **kwargs))

        if self.summary_chain is None:
            if self.summary_prompt is not None:
                # Use coordinator as summary agent
                self.state = f"{self.planner_agent.card.name} finalizing"
                message.query = self.summary_prompt
                message.context = {self.summary_steps_key: sub_task_result}
                message = self.planner_agent.execute(message, **kwargs)
            # if no summary prompt is provided, just return the result from workers
        else:
            self.state = "finalizing"
            message.context = {self.summary_steps_key: sub_task_result}
            message = self.summary_chain.invoke(message, **kwargs)

        self.state = "idle"
        message.origin = self.card.name
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, [self.planner_agent, *list(self.workers.values())], "sequential"
        )


class DebateAgent(BaseAgent):
    agents: List[BaseAgent] = Field(
        ..., description="""The list of agents participate in the conversation"""
    )
    pick_strategy: (
        Literal["round_robin", "random", "simultaneous"]
        | Callable[[List[BaseAgent]], BaseAgent]
    ) = Field(
        "round_robin", description="The strategy to pick the agent to run the next turn"
    )
    max_turns: int = Field(5, description="The maximum number of debate turns to run")
    should_stop: Callable[[AgentMessage], bool] | None = Field(
        None, description="The condition to stop the debate"
    )

    def __init__(
        self,
        card: AgentCard,
        agents: List[BaseAgent],
        pick_strategy: Literal["round_robin", "random", "simultaneous"]
        | Callable[[List[BaseAgent]], BaseAgent] = "round_robin",
        max_turns: int = 5,
        should_stop: Callable[[AgentMessage], bool] | None = None,
        state_change_callback: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            state_change_callback=state_change_callback, card=card, **kwargs
        )
        self.agents = agents
        self.pick_strategy = pick_strategy
        self.max_turns = max_turns
        self.should_stop = should_stop
        for agent in self.agents:
            agent.state_change_callback = self._child_state_observer

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        n = 0
        message.responses = []

        # All response turns are stored in the message.responses
        while n < self.max_turns:
            if self.pick_strategy == "round_robin":
                next_turn_agent = self.agents[n % len(self.agents)]
            elif self.pick_strategy == "simultaneous":
                self.state = f"turn {n}: all agents running"
                # TODO: parallelize the execution
                current_turn_messages = [
                    agent.execute(message, **kwargs) for agent in self.agents
                ]
                message.responses.extend(  # type: ignore
                    [
                        (msg.origin or "", msg.response or "")
                        for msg in current_turn_messages
                    ]
                )
                n += len(self.agents)
            elif self.pick_strategy == "random":
                next_turn_agent = random.choice(self.agents)
            elif isinstance(self.pick_strategy, Callable):
                next_turn_agent = self.pick_strategy(self.agents)
            else:
                raise ValueError(
                    "pick_strategy must be one of 'round_robin', 'random', 'simutaneous' or a function"
                )

            if self.pick_strategy != "simultaneous":
                self.state = f"turn {n}: agent {next_turn_agent.card.name} running"  # type: ignore
                message = next_turn_agent.execute(message, **kwargs)  # type: ignore
                message.responses.append((next_turn_agent.card.name, message.response))  # type: ignore
                n += 1

            if message.execution_result != "success" or (
                self.should_stop and self.should_stop(message)
            ):
                break

        self.state = "idle"
        message.origin = self.card.name
        message.execution_result = "success"
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self,
            self.agents,
            "parallel" if self.pick_strategy == "simultaneous" else "sequential",
        )


class VotingAgent(BaseAgent):
    agents: List[BaseAgent] = Field(
        ..., description="""The list of agents participate in the voting"""
    )
    voting_method: Literal["agent_forest", "llm_score", "majority_vote"] = Field(
        "agent_forest", description="The voting method to use"
    )
    voting_prompt: str | None = Field(
        None,
        description="The prompt to use for the voting. This parameter is used when voting_method is 'llm_score' or 'majority_vote'",
    )
    get_score_func: Callable[[str], float] = Field(
        float,
        description="""The function to use to convert LLM response to score, which is required only when voting_method is 'llm_score'.
        Note that LLM may generate reponse that could not convert to score.
        In these cases, there are various ways to handle.
        For example, we just ignore it and assign the score to 0""",
    )

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        if message.responses is None:
            message.responses = []
        if self.voting_method == "agent_forest":
            # https://arxiv.org/pdf/2402.05120
            scores = {agent.card.name: 0.0 for agent in self.agents}
            message_map = {name: msg for name, msg in message.responses}
            for agent_name, msg in message.responses:
                total_score = 0
                for other_agent in self.agents:
                    if agent_name == other_agent.card.name:
                        continue
                    total_score += sentence_bleu(
                        msg, [message_map[other_agent.card.name]], lowercase=True
                    ).score
                scores[agent_name] = total_score
            highest_score_agent = max(scores, key=scores.get)  # type: ignore
            message.response = message_map[highest_score_agent]

        elif self.voting_method == "llm_score":
            if self.voting_prompt is None:
                raise ValueError(
                    "voting_prompt is required for llm_score voting method"
                )
            message.query = self.voting_prompt
            scores = {agent.card.name: 0.0 for agent in self.agents}
            message_map = {name: msg for name, msg in message.responses}
            for response in message.responses:
                total_score = 0
                for agent in self.agents:
                    if response[0] == agent.card.name:
                        continue

                    score = agent.execute(message, **kwargs)
                    try:
                        total_score += self.get_score_func(str(score.response))
                    except ValueError as e:
                        # unable to parse the score
                        pass
                scores[response[0]] = total_score
                # The highest or average score will return the same response
                # Additional note: because of the score, we even can rank all candidate responses and output a leaderboard.
            highest_score_agent = max(scores, key=scores.get)  # type: ignore
            message.response = message_map[highest_score_agent]

        elif self.voting_method == "majority_vote":
            raise NotImplementedError

        else:
            raise ValueError(
                "voting_method must be one of 'agent_forest', 'llm_score', 'majority_vote'"
            )

        self.state = "idle"
        message.origin = self.card.name
        message.execution_result = "success"
        return message

    def _set_composed_state(self) -> None:
        self._composed_state = BaseAgent.build_composed_state(
            self, self.agents, "sequential"
        )
