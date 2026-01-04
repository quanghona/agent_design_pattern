import abc
from collections.abc import Callable, Sequence
import copy
from typing import Literal, Tuple
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
    _scorer: Callable[[BaseLLMChain, str, DataSet], float] = PrivateAttr()

    base_chain: BaseLLMChain = Field(
        ..., description="The base LLM chain to use for SEE"
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
    feedback_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Feedback operator"
    )
    semantic_chain: BaseLLMChain = Field(
        ..., description="The LLM chain to use for Semantic operator"
    )

    dev_set: DataSet = Field(
        ...,
        description="The dataset for evaluate prompt. This is D_dev in the algorithm",
    )
    init_data: DataSet | str = Field(
        ...,
        description="""The intial data for phase 0. There are two types of initialzation:
        - See-io-pair: provide a set of input-output pairs. SEE apply Lamarckian to generate prompts
        - SEE-example: SEE take a intial prompt and use Semantic to generate new prompts
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
    crossover_with_distinct: bool = Field(
        default=False,
        description="Whether to use Crossover with distinct to generate new prompt",
    )

    def __init__(
        self,
        scorer: Callable[[BaseLLMChain, str, DataSet], float]
        | Literal["hamming", "levenshtein", "cosine"] = "hamming",
        **kwargs,
    ):
        warnings.warn(
            "This class is only for experimental use. It is not recommended for production use."
        )

        super().__init__(**kwargs)
        if scorer == "hamming":
            self._scorer = SEEPromptAugmenter._hamming_scorer
        elif scorer == "levenshtein":
            self._scorer = SEEPromptAugmenter._levenshtein_scorer
        elif scorer == "cosine":
            self._scorer = SEEPromptAugmenter._cosine_scorer
        elif isinstance(scorer, Callable):
            self._scorer = scorer

    @classmethod
    def _hamming_scorer(
        cls, chain: BaseLLMChain, prompt: str, dataset: DataSet
    ) -> float:
        return 0.0

    @classmethod
    def _levenshtein_scorer(
        cls, chain: BaseLLMChain, prompt: str, dataset: DataSet
    ) -> float:
        return 0.0

    @classmethod
    def _cosine_scorer(
        cls, chain: BaseLLMChain, prompt: str, dataset: DataSet
    ) -> float:
        return 0.0

    # TODO: extend algorithm to multimodal data
    def lamarckian(
        self,
        pairs: DataSet,
        **kwargs,
    ) -> str:
        msg = self.lamarckian_chain.invoke(message, pairs=pairs, **kwargs)
        return msg.responses[-1][1]

    def eda(self, candidates: Sequence[str], **kwargs) -> str:
        pass

    def crossover(self, parents: Sequence[str], **kwargs) -> str:
        # Choose 2 parents from a list of prompts
        pass

    def feedback(self, candidate: str, **kwargs) -> str:
        pass

    def semantic(self, candidate: str, **kwargs) -> str:
        message = AgentMessage(query="", context={"candidate": candidate})
        message = self.semantic_chain(message, **kwargs)
        return message.responses[-1][1]

    def augment(self, message: AgentMessage, **kwargs) -> AgentMessage:
        P = []  # prompt pool
        S = []  # corresponding scores

        # phase 0: global initialization
        # use lamarckian or sematic to diverse initial prompt
        if isinstance(self.init_data, str):
            for _ in range(self.pool_size_0):
                prompt = self.semantic(self.init_data)
                P.append(prompt)
                S.append(self._scorer(self.base_chain, prompt, self.dev_set))
        else:
            for _ in range(self.pool_size_0):
                prompt = self.lamarckian(self.init_data)
                P.append(prompt)
                S.append(self._scorer(self.base_chain, prompt, self.dev_set))

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
            score = self._scorer(self.base_chain, new_prompt, self.dev_set)
            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            t = (score - S[k]) / S[k]
            if score > S[k]:
                P[k] = new_prompt
                S[k] = score
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
            score = self._scorer(self.base_chain, new_prompt, self.dev_set)
            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            t = (score - S[k]) / S[k]
            if score > S[k]:
                P[k] = new_prompt
                S[k] = score

            k += 1

        max_score_index = np.argmax(S)
        message.query = P[max_score_index]
        return message
