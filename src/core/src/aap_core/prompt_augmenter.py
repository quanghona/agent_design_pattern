import abc
from collections.abc import Callable, Sequence
import copy
from typing import Concatenate, Dict, Literal, Tuple
import warnings

from aap_core.chain import BaseLLMChain
from pydantic import Field, PrivateAttr, field_validator

from aap_core.retriever import BaseRetriever

# import toon_format
from .types import AgentMessage, BaseChain
import numpy as np


class BasePromptAugmenter(BaseChain):
    """A base class to enhance / rewrite the prompt.

    There are two types of prompt augmenter:
    - Data augmenter: Give more context to the prompt by adding external data, either by using files (CSV, JSON, Markdown, etc.), database (SQL,...) or more advanced techniques like RAG, web search.
    - Structure augmenter: Rewrite / refine the prompt partially or entirely.
    """

    # TODO: handle case where single augmenter contains multiple retrievers
    loop: int | Callable[[AgentMessage], bool] | None = Field(
        default=None,
        description="The loop, either by number of times or by stop condition",
    )
    retriever: BaseRetriever | None = Field(default=None, description="The retriever")

    async def acall(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    @abc.abstractmethod
    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return self(message, **kwargs)

    def call(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if self.retriever:
            message = self.retriever(message, **kwargs)
        return self.augment(message, **kwargs)

    def __call__(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if isinstance(self.loop, int):
            for _ in range(self.loop):
                message = self.call(message, **kwargs)
            return message
        elif callable(self.loop):
            while self.loop(message):
                message = self.call(message, **kwargs)
            return message
        else:
            return self.call(message, **kwargs)


class IdentityPromptAugmenter(BasePromptAugmenter):
    """A prompt enhancer that does nothing.
    This serves as a default prompt enhancer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = None
        self.retriever = None

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        return message


class SimplePromptAugmenter(BasePromptAugmenter):
    """A prompt enhancer that uses a template to augment the prompt.
    Specifically, it simply concatenate the original prompt with the context data,
    using provided format.

    This augmenter mainly used for simple scenarios such as tabular data concatenation or naive RAG.
    """

    format: str = Field(
        ...,
        description="""
        The format of the prompt.
        The format must contain at least {query} and {context}.
        Other parameters can be used and parsed""",
    )
    data_key: str = Field(
        default="context.data", description="The key to the context data in the message"
    )

    @field_validator("format")
    @classmethod
    def check_starts_with_prompt_and_data(cls, v: str) -> str:
        if "{query}" not in v or "{data}" not in v:
            raise ValueError("The format must contain at least {query} and {data}")
        return v

    @field_validator("data_key")
    @classmethod
    def check_starts_with_prefix(cls, v: str) -> str:
        if not v.startswith("context."):
            raise ValueError("data_key must start with 'context.'")
        return v

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        if message.context is None:
            raise ValueError("Message context is None")
        elif self.data_key not in message.context:
            raise ValueError(f"Message context does not contain {self.data_key}")
        context_key = self.data_key.replace("context.", "")
        message.query = self.format.format(
            query=message.query, data=message.context[context_key], **kwargs
        )
        return message


class GEPAPromptAugmenter(BasePromptAugmenter):
    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        raise NotImplementedError


class SEEPromptAugmenter(BasePromptAugmenter):
    """SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization
    https://arxiv.org/abs/2402.11347

    SEE uses LLM operators to perform generation and variation. There are 5 operators introduced in the work:
    - Lamarckian: reverse engineering by generating prompt from a set of input-output pairs
    - EDA (Estimation of Distribution): takes in a group of candidates and otuputs a new candidat by studying the input group.
    - Crossover: mixing the traits of both parents and generates a new candidate
    - Feedback: use 2 agents - Examiner and Improver to generate new candidate
    - Semantic: modifies the candidate lexically while preserving its semantic meaning

    The SEE framework contains 4 phases:
    - Phase 0: Global initialization
    - Phase 1: Local feedback operation
    - Phase 2: Global fusion operation
    - Phase 3: Local semantic operation

    As stated in the paper, SEE needs multiple iterations and relatively large amount of API call, which might be inefficient for large-scale production.
    So this is only for experimental use.
    """

    DataSet = Sequence[Tuple[str, str]]
    PerformanceTuple = Tuple[Sequence[float], float]
    _scorer: Callable[
        Concatenate[
            BaseLLMChain, str, Sequence[str], Sequence[PerformanceTuple], DataSet, ...
        ],
        PerformanceTuple | None,
    ] = PrivateAttr()
    _scorer_args: Dict = PrivateAttr()

    base_chain: BaseLLMChain = Field(
        ..., description="The base LLM chain used by SEE to generate result"
    )
    lamarckian_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Lamarckian operator"
    )
    eda_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for EDA operator"
    )
    crossover_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Crossover operator"
    )
    examiner_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Feedback operator"
    )
    improver_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Feedback operator"
    )
    semantic_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Semantic operator"
    )

    dev_set: DataSet = Field(
        ...,
        description="The dataset for evaluate prompt. This is D_dev in the algorithm",
        min_length=1,
    )
    init_data: DataSet | str = Field(
        ...,
        description="""The intial data for phase 0. There are two types of initialzation:
        - See-io-pair: provide a set of input-output pairs. SEE apply Lamarckian to generate prompts
        - SEE-example: SEE take a intial prompt and use Semantic to generate new prompts
        """,
    )

    lamarckian_message: AgentMessage | None = Field(
        default=None,
        description="""The prompt use for Lamarckian operator.
        If not provided, the default prompt template will be constructed,
        The default key for pairs is {context.pairs}.
        """,
    )
    eda_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for EDA operator.
    If not provided, the default prompt template will be constructed,
    The default key for candidates is {context.candidates}.
    """,
    )
    crossover_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for Crossover operator.
    If not provided, the default prompt template will be constructed,
    The default key for parents is {context.parents}.
""",
    )
    examiner_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for examine candidate, for feedback operator
    If not provided, the default prompt template will be constructed
    the default key for wrong cases is {context.wrong_cases}
""",
    )
    improver_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use to improve the existing candidate, for feedback operator.
    If not provided, the default prompt template will be constructed,
    The default key for feedback is {context.feedback}
""",
    )
    semantic_message: AgentMessage | None = Field(
        default=None,
        description="""
    The prompt use for Semantic operator.
    If not provided, the default prompt template will be constructed,
    The default key for candidate is {context.candidate}.
    """,
    )

    pool_size_0: int = Field(
        default=15,
        description="Pool size of phase 0: Global initialization. Mark as n_0 in the algorithm",
    )
    pool_size_1: int = Field(
        default=5,
        description="Pool size of phase 1: Local feedback. Mark as n_1 in the algorithm",
    )
    pool_size_2: int = Field(
        default=5,
        description="Pool size of phase 2: Global fusion. Mark as n_2 in the algorithm",
    )
    pool_size_3: int = Field(
        default=5,
        description="Pool size of phase 3: Local semantic. Mark as n_3 in the algorithm",
    )
    tolerance_1: int = Field(
        default=1,
        description="Tolerance for phase 1: Local feedback, aka for feedback operator. This is marked as K_1 in the algorithm",
    )
    tolerance_2: int = Field(
        default=1,
        description="Tolerance for phase 2: Global fusion, aka for semantic operator. This is marked as K_2 in the algorithm",
    )
    tolerance_3: int = Field(
        default=4,
        description="Tolerance for phase 3: Local semantic, aka for EDA and crossover operator. This is marked as K_3 in the algorithm",
    )
    performance_gain_threshold: float = Field(
        default=0.01, description="Performance gain threshold"
    )

    def __init__(
        self,
        scorer: Callable[
            Concatenate[
                BaseLLMChain,
                str,
                Sequence[str],
                Sequence[PerformanceTuple],
                DataSet,
                ...,
            ],
            PerformanceTuple | None,
        ]
        | Literal["hamming"] = "hamming",
        scorer_args: Dict = {},
        **kwargs,
    ):
        warnings.warn(
            "This class is only for experimental use. It is not recommended for production use."
        )

        super().__init__(**kwargs)
        if scorer == "hamming":
            self._scorer = SEEPromptAugmenter._hamming_scorer
        # elif scorer == "levenshtein":
        #     self._scorer = SEEPromptAugmenter._levenshtein_scorer
        # elif scorer == "cosine":
        #     self._scorer = SEEPromptAugmenter._cosine_scorer
        elif isinstance(scorer, Callable):
            self._scorer = scorer
        else:
            raise ValueError("scorer not supported")
        self._scorer_args = scorer_args

    @classmethod
    def score(
        cls, chain: BaseLLMChain, prompt: str, dataset: DataSet
    ) -> PerformanceTuple:
        performance_vector = []
        for data in dataset:
            query = f"""{prompt}
            Question:
            {data[0]}
            Answer:"""
            message = chain.invoke(AgentMessage(query=query))
            # TODO: diverse options for selecting comparasion method
            performance_vector.append(message.responses[-1][1] == data[1])
        return performance_vector, sum(performance_vector) / len(performance_vector)

    @classmethod
    def _hamming_scorer(
        cls,
        chain: BaseLLMChain,
        prompt: str,
        pool: Sequence[str],
        performance_pool: Sequence[PerformanceTuple],
        dataset: DataSet,
        distance_threshold: int = 2,
    ) -> PerformanceTuple | None:
        perf_vec, score = SEEPromptAugmenter.score(chain, prompt, dataset)
        # can only check with lowest score candidate
        min_dist = min(
            np.count_nonzero(np.array(perf_vec) != np.array(p))
            for p in performance_pool
        )  # hamming distance
        # Found similar candidate -> NOT use this prompt
        return None if min_dist < distance_threshold else (perf_vec, score)


    # TODO: extend algorithm to multimodal data
    def lamarckian(self, pairs: DataSet, **kwargs) -> str:
        dataset = "\n\n".join(
            [f"Input: {pair[0]}\nOutput: {pair[1]}" for pair in pairs]
        )
        if self.lamarckian_message is None:
            # The default prompt in the paper
            message = AgentMessage(
                query="""I gave a friend an instruction and some input. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

## Example ##
{context.pairs}

The instruction was:
""",
                context={"pairs": dataset},
            )
        else:
            message = self.lamarckian_message
            if message.context is None:
                message.context = {"pairs": dataset}
            else:
                message.context["pairs"] = dataset
        msg = self.lamarckian_chain.invoke(message, **kwargs)
        return msg.responses[-1][1]

    def eda(self, candidates: Sequence[str], **kwargs) -> str:
        # TODO: Choose parents from a list of prompts
        cand_str = "\n\n".join(candidates)
        if self.eda_message is None:
            message = AgentMessage(
                query="""You are a mutator. Given a series of prompts, your task is to generate another prompt
with the same semantic meaning and intentions.

## Existing Prompts ##
{context.candidates}

The newly mutated prompt is:""",
                context={"candidates": cand_str},
            )
        else:
            message = self.eda_message
            if message.context is None:
                message.context = {"candidates": cand_str}
            else:
                message.context["candidates"] = cand_str
        message = self.eda_chain(message, **kwargs)
        return message.responses[-1][1]

    def crossover(self, parents: Sequence[str], **kwargs) -> str:
        # TODO: Choose parents from a list of prompts
        chosen_parent = []
        parent_prompt = ""
        for parent in parents:
            chosen_parent.append(parent)
            parent_prompt += f"Parent prompt {len(chosen_parent)}: {parent}\n"
        if self.crossover_message is None:
            message = AgentMessage(
                query="""You are a mutator who is familiar with the concept of cross-over in genetic algorithm,
namely combining the genetic information of two parents to generate new offspring.
Given two parent prompts, you will perform a cross-over to generate an offspring
prompt that covers the same semantic meaning as both parents.
# Example
Parent prompt 1: Now you are a categorizer, your mission is to ascertain the
sentiment of the provided text, either favorable or unfavorable
Parent prompt 2: Assign a sentiment label to the given sentence from ['negative', 'positive'] and return only the label without any other text.
Offspring prompt: Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from ['negative', 'positive'].

## Given ##
{context.parents}
Offspring prompt:
""",
                context={"parents": parent_prompt},
            )
        else:
            message = self.crossover_message
            if message.context is None:
                message.context = {"parents": parent_prompt}
            else:
                message.context["parents"] = parent_prompt
        message = self.crossover_chain(message, **kwargs)
        return message.responses[-1][1]

    def feedback(self, candidate: str, **kwargs) -> str:
        # TODO: put wrong cases to prompt
        if self.examiner_message is None:
            message = AgentMessage(
                query="""You are a quick improver. Given an existing prompt and a series of cases where it
made mistakes. Look through each case carefully and identify what is causing the
mistakes. Based on these observations, output ways to improve the prompts based
on the mistakes.
## Existing Prompt ##
{context.candidate}
## Cases where it gets wrong:##
{context.wrong_cases}
ways to improve the existing prompt based on observations of the mistakes in the
cases above are:
""",
                context={"candidate": candidate, "wrong_cases": ""},
            )
        else:
            message = self.examiner_message
            if message.context is None:
                message.context = {"candidate": candidate, "wrong_cases": ""}
            else:
                message.context["candidate"] = candidate
                message.context["wrong_cases"] = ""
        message = self.examiner_chain(message, **kwargs)

        feedback_msg = message.responses[-1][1]
        if self.improver_message is None:
            message = AgentMessage(
                query="""You are a quick improver. Given an existing prompt and feedback on how it should
improve. Create an improved version based on the feedback.
## Existing Prompt ##
{context.candidate}
## Feedback ##
{context.feedback}
## Improved Prompt ##""",
                context={"candidate": candidate, "feedback": feedback_msg},
            )
        else:
            message = self.improver_message
            if message.context is None:
                message.context = {"candidate": candidate, "feedback": feedback_msg}
            else:
                message.context["candidate"] = candidate
                message.context["feedback"] = feedback_msg
        message = self.improver_chain(message, **kwargs)
        return message.responses[-1][1]

    def semantic(self, candidate: str, **kwargs) -> str:
        if self.semantic_message is None:
            message = AgentMessage(
                query="""You are a mutator. Given a prompt, your task is to generate another prompt with the
same semantic meaning and intentions.
# Example:
current prompt: Your mission is to ascertain the sentiment of the provided text and
assign a sentiment label from ['negative', 'positive'].
mutated prompt: Determine the sentiment of the given sentence and assign a label
from ['negative', 'positive'].

## Given ##
current prompt: {context.candidate}
mutated prompt:
""",
                context={"candidate": candidate},
            )
        else:
            message = self.semantic_message
            if message.context is None:
                message.context = {"candidate": candidate}
            else:
                message.context["candidate"] = candidate
        message = self.semantic_chain(message, **kwargs)
        return message.responses[-1][1]

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        P = []  # prompt pool
        S = []  # corresponding performance scores on dev set

        # phase 0: global initialization
        # use lamarckian or sematic to diverse initial prompt
        if isinstance(self.init_data, str):
            while len(P) < self.pool_size_0:
                prompt = self.semantic(self.init_data)
                perf = self._scorer(self.base_chain, prompt, P, S, self.dev_set)
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)
        else:
            while len(P) < self.pool_size_0:
                prompt = self.lamarckian(self.init_data)
                perf = self._scorer(self.base_chain, prompt, P, S, self.dev_set)
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)

        # phase 1: local feedback operation
        # use feedback to generate new prompt
        sorted_indices = np.argsort(S)
        P_t = [P[i] for i in sorted_indices]
        S_t = sorted(S, reverse=True)
        P_t = P_t[: self.pool_size_1]
        S_t = S_t[: self.pool_size_1]
        t = 0
        k = 0
        while t < self.performance_gain_threshold or k <= self.tolerance_1:
            prompt = P[k]
            new_prompt = self.feedback(prompt)
            perf = self._scorer(self.base_chain, prompt, P, S, self.dev_set)
            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            if perf is not None and perf[1] > S[k][1]:
                P[k] = new_prompt
                S[k] = perf
                t = (perf[1] - S[k][1]) / S[k][1]
            k += 1

        # phase 2: global fusion operation
        # use eda and crossover to fusion existing prompt and generate new prompt
        sorted_indices = np.argsort(S)
        P_t = [P[i] for i in sorted_indices]
        S_t = sorted(S, reverse=True)
        avg_score = np.mean(S)
        t = 0  # last phase performance
        k = 0
        while t < self.performance_gain_threshold or k <= self.tolerance_2:
            # TODO: choose 2 parents
            # According to paper, each operator tolerance is self.tolerance_2
            # -> total tolerance for this phase is 2 * self.tolerance_2. Experiment in the paper chose 4 for each operator as default value
            new_prompt = self.eda(P_t)
            new_prompt = self.crossover(P_t)
            # TODO: Add new prompt only when it does not have a similarity score
            # over a threshold with any other candidate that is already in the subset.
            # The subset will be randomized before
            # feeding into the LLM agent so the candidate’s performance does not dictate its order.

            # Performance gain is defined as whether the average performance of the pool is improved
            # Note: we calculate the average score using highest score members and limited by the pool size
            sorted_indices = np.argsort(S)
            P_t = [P[i] for i in sorted_indices][: self.pool_size_2]
            S_t = sorted(S_t, reverse=True)[: self.pool_size_2]
            new_avg_score = np.mean(S_t)
            t = (new_avg_score - avg_score) / avg_score
            k += 1

        # phase 3: local semantic operation
        # use semantic to generate new prompts
        P_t = P_t[: self.pool_size_3]
        S_t = S_t[: self.pool_size_3]
        t = 0  # last phase performance
        k = 0
        while t < self.performance_gain_threshold or k <= self.tolerance_3:
            prompt = P[k]
            new_prompt = self.semantic(prompt)
            perf = self._scorer(self.base_chain, prompt, P, S, self.dev_set)
            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            if perf is not None and perf[1] > S[k][1]:
                P[k] = new_prompt
                S[k] = perf
                t = (perf[1] - S[k][1]) / S[k][1]
            k += 1

        max_score_index = np.argmax(S)
        message.query = P[max_score_index]
        return message
