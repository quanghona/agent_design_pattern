import abc
from collections.abc import Callable, Sequence
from typing import Concatenate, Dict, List, Literal, Tuple
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
    """# SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization
    https://arxiv.org/abs/2402.11347

    This is a self implementation that adapt to ours aap framework with additional extensions.
    The logic may not be identical to the original work.

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

    Regarding prompts for operators, we have 3 ways to distribute prompt into user prompt and system prompt:
    - Put all prompt content in the system prompt template and user prompt is empty
    - The system prompt is the fixed instruction, and the template and examples are in the user prompt. This is the recommended implementation
    - The system prompt is empty and all the prompt content stay in the user prompt. This approach has highest flexibility but only suitable for development

    To maximize the effect of operators, we should adjust the temperature of LLM accordingly.
    Push the temperature higher for exploring phases (0 and 2), 3 global operators: Lamarckian, EDA, Crossover.
    Lower the temperature for exploiting phases (1 and 3), 2 local operators: Feedback, Semantic.
    The adjustment stays inside the logic of the chains, which is outside of this SEE class

    Note that we could use the same or different LLMs for different operators and the baseline.

    As stated in the paper, SEE needs multiple iterations and relatively large amount of API call, which might be inefficient for large-scale production.
    So this is only for experimental use.

    Following our framework, in the prompt template, the keyword need to be prefix with 'context.' and the additional
    data need to store in 'context' field of the AgentMessage object
    """

    DataSet = Sequence[Tuple[str, str]]
    PerformanceTuple = Tuple[Sequence[float], float]

    _scorer: Callable[
        Concatenate[
            BaseLLMChain, str, Sequence[str], Sequence[PerformanceTuple], DataSet, ...
        ],
        PerformanceTuple | None,
    ] = PrivateAttr()
    _dist_func: Callable[[Sequence[float], Sequence[float]], float] = PrivateAttr()
    _scorer_args: Dict = PrivateAttr()
    _eval_method: Callable[[str, str], bool] = PrivateAttr()

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
        description="Pool size of phase 0: Global initialization. Marked as n_0 in the algorithm",
    )
    pool_size_1: int = Field(
        default=5,
        description="Pool size of phase 1: Local feedback. Marked as n_1 in the algorithm",
    )
    pool_size_2: int = Field(
        default=5,
        description="Pool size of phase 2: Global fusion. Marked as n_2 in the algorithm",
    )
    pool_size_3: int = Field(
        default=5,
        description="Pool size of phase 3: Local semantic. Marked as n_3 in the algorithm",
    )
    tolerance_1: int = Field(
        default=1,
        description="Tolerance for phase 1: Local feedback, aka for feedback operator. This is marked as K_1 in the algorithm",
    )
    tolerance_2: int = Field(
        default=8,
        description="""Tolerance for phase 2: Global fusion, aka for EDA and crossover operator. This is marked as K_2 in the algorithm.
        Note that this is tolerance for whose phase, which is total tolerance for 2 operators""",
    )
    tolerance_3: int = Field(
        default=1,
        description="Tolerance for phase 3: Local semantic, aka for semantic operator. This is marked as K_3 in the algorithm",
    )
    performance_gain_threshold: float = Field(
        default=0.01, description="Performance gain threshold. In percent scale"
    )
    num_crossover_parents: int = Field(
        default=2,
        description="""Number of parents to use for exploring operator such as EDA and Crossover.
        Setting value to -1 will use all candidates in the prompt pool.""",
    )
    num_eda_parents: int = Field(
        default=-1,
        description="""Number of parents to use for exploring operator such as EDA and Crossover
        Setting value to -1 will use all candidates in the prompt pool.""",
    )
    eda_with_index: bool = Field(
        default=False,
        description="Whether to rank candidates",
    )
    crossover_with_distinct: bool = Field(
        default=False,
        description="Whether to consider the diversity in parents",
    )
    num_feedback_wrongcases: int = Field(
        default=3,
        description="Number of wrong cases to include in the prompt for feedback operator",
        gt=0,
    )

    def __init__(
        self,
        scorer: Literal["hamming"]
        | Callable[
            Concatenate[
                BaseLLMChain,
                str,
                Sequence[str],
                Sequence[PerformanceTuple],
                DataSet,
                ...,
            ],
            PerformanceTuple | None,
        ] = "hamming",
        dist_func: Callable[[Sequence[float], Sequence[float]], float] | None = None,
        scorer_args: Dict = {},
        eval_method: Literal["exact", "include"] | Callable[[str, str], bool] = "exact",
        **kwargs,
    ):
        warnings.warn(
            "This class is only for experimental use. It is not recommended for production use."
        )

        super().__init__(**kwargs)
        if scorer == "hamming":
            self._scorer = self._hamming_scorer
            self._dist_func = SEEPromptAugmenter._hamming_distance
        # elif scorer == "levenshtein":
        #     self._scorer = SEEPromptAugmenter._levenshtein_scorer
        # elif scorer == "cosine":
        #     self._scorer = SEEPromptAugmenter._cosine_scorer
        elif isinstance(scorer, Callable) and dist_func is not None:
            self._scorer = scorer
            self._dist_func = dist_func
        else:
            raise ValueError("scorer not supported")
        self._scorer_args = scorer_args

        if eval_method == "exact":
            self._eval_method = lambda x, y: x == y
        elif eval_method == "include":
            self._eval_method = lambda x, y: x in y
        elif isinstance(eval_method, Callable):
            self._eval_method = eval_method
        else:
            raise ValueError("eval_method not supported")

    def score(
        self,
        prompt: str,
        dataset: DataSet,
    ) -> PerformanceTuple:
        performance_vector = []
        for data in dataset:
            query = f"""{prompt}
            Question:
            {data[0]}
            Answer:"""
            message = self.base_chain.invoke(AgentMessage(query=query))
            performance_vector.append(
                int(self._eval_method(data[1], message.responses[-1][1]))
            )
        return performance_vector, sum(performance_vector) / len(performance_vector)

    @classmethod
    def _hamming_distance(cls, v1: Sequence[float], v2: Sequence[float]) -> float:
        return np.count_nonzero(np.array(v1) != np.array(v2))

    def _hamming_scorer(
        self,
        chain: BaseLLMChain,
        prompt: str,
        pool: Sequence[str],
        performance_pool: Sequence[PerformanceTuple],
        dataset: DataSet,
        distance_threshold: int = 2,
    ) -> PerformanceTuple | None:
        perf_vec, score = self.score(prompt, dataset)
        # can only check with lowest score candidate
        min_dist = min(
            SEEPromptAugmenter._hamming_distance(perf_vec, p[0])
            for p in performance_pool
        )  # hamming distance
        # Found similar candidate -> NOT use this prompt
        return None if min_dist < distance_threshold else (perf_vec, score)


    @classmethod
    def _sort_pool(
        cls, P_t: list[str], S_t: list[PerformanceTuple], pool_size: int
    ) -> Tuple[List[str], List[PerformanceTuple]]:
        sorted_indices = np.argsort(S_t)
        P_t = [P_t[i] for i in sorted_indices]
        S_t = sorted(S_t, reverse=True)
        return P_t[:pool_size], S_t[:pool_size]

    # TODO: extend algorithm to multimodal data
    def lamarckian(self, pairs: DataSet, **kwargs) -> str:
        """
        Generate new prompt using Lamarckian operator
        This method works with corresponding lamarckian_message and lamarckian_chain attributes.
        In the prompt in lamarckian_message, it must contains key 'context.pairs' to fetch data from provided dataset.

        Args:
            pairs (DataSet): input-output pairs of the data
            **kwargs: arguments for lamarckian chain

        Returns:
            str: the generated prompt
        """
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
        """
        Generate new prompt using EDA operator
        This method works with corresponding eda_message and eda_chain attributes.

        In the prompt in eda_message, it must contains key 'context.candidates' to fetch candidates.
        We can control number of candidates by setting num_eda_parents.
        If we not apply indexing, the candiadates will be shuffled before fetch into prompt message

        Args:
            candidates (Sequence[str]): list of candidate prompts
            **kwargs: arguments for eda chain

        Returns:
            str: the generated prompt
        """
        num_parents = (
            len(candidates)
            if self.num_eda_parents < 0
            else min(len(candidates), self.num_eda_parents)
        )
        indices = np.random.choice(len(candidates), num_parents, replace=False)
        if self.eda_with_index:
            indices = np.sort(indices)
        parents = [candidates[i] for i in list(map(int, indices))]
        cand_str = "\n\n".join(parents)
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

    @classmethod
    def max_vector_distance_subarray(
        cls,
        S: Sequence[Sequence[float]],
        k: int,
        dist_func: Callable[[Sequence[float], Sequence[float]], float],
    ) -> Sequence[int]:
        if k > len(S):
            raise ValueError("k cannot be greater than the length of S.")
        if k < 2:
            return list(range(k))

        selected_indices = [int(np.random.randint(0, len(S)))]

        # Greedily add points until we reach size k
        while len(selected_indices) < k:
            best_next_idx = -1
            max_total_extra_dist = -1

            for i in range(len(S)):
                if i in selected_indices:
                    continue

                # Calculate how much this point adds to the total pairwise distance
                current_extra_dist = sum(
                    dist_func(S[i], S[idx]) for idx in selected_indices
                )

                if current_extra_dist > max_total_extra_dist:
                    max_total_extra_dist = current_extra_dist
                    best_next_idx = i

            selected_indices.append(best_next_idx)

        return selected_indices

    def crossover(
        self, P: Sequence[str], S: Sequence[PerformanceTuple], **kwargs
    ) -> str:
        """
        Generate new prompt using Crossover mutation
        This method works with corresponding crossover_message and crossover_chain attributes.

        In the prompt in crossover_message, it must contains key 'context.parents' to fetch parents.
        We can control number of parents by setting num_crossover_parents.

        In the original work, the operator only accepts 2 parents. We generalize it
        to accept any number of parents. If diversity is not considered, we choose the best k parents.
        Otherwise we formulate the problem as below:

        Given a set of n parents P with its corresponding performance S, a target number of selected parents k.
        Find a subset of k parents that maximize the total pairwise distance between performance vector of selected parents.

        Hence, the definition 4 (Crossover Operator - CR) in the paper is generalized:
        Crossover generates a new candidate based on k parents. It is a function
        operator O_C that perform O_C(P_k, L) = p' where P_k is a subset of P,
        p' is the generated prompt that hopefully hold features from prompts in P_k.
        If P_k is chosen from P so that it maximizes \sum_{i < j}^{P_k}d(p_i, p_j), we call it Crossover + Distinct (CR+D)

        If k = 2, it fallbacks to the original work
        We gather P_k by greedily adding points until we reach size k. First we
        find 2 initial points that have the largest distance among all pairs.
        Then for the rest points, we iteratively find the point that have max distance
        with the current set. The implementation of the algorithm is in `max_vector_distance_subarray` method

        Selected parents will be shuffled before fetch into prompt message

        Args:
            P (Sequence[str]): input prompts
            S (Sequence[PerformanceTuple]): corresponding performance vectors.
            **kwargs: arguments for crossover chain

        Returns:
            str: the generated prompt
        """
        num_parents = (
            len(P)
            if self.num_crossover_parents < 0
            else min(len(P), self.num_crossover_parents)
        )

        if self.crossover_with_distinct:
            indices = SEEPromptAugmenter.max_vector_distance_subarray(
                [s[0] for s in S], num_parents, self._dist_func
            )
            parents = [P[i] for i in indices]
        else:
            # Choose the best parents
            parents = P[:num_parents]

        np.random.shuffle(parents)
        parent_prompt = ""
        for parent in parents:
            parent_prompt += f"Parent prompt {len(parents)}: {parent}\n"
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

    def feedback(self, candidate: str, performance: PerformanceTuple, **kwargs) -> str:
        """
        Generate an improved version of the prompt based on the feedback from the examiner.

        Args:
            candidate (str): the original prompt
            **kwargs: arguments for Examiner and Improver chains

        Returns:
            str: the generated prompt
        """
        perf_vec, _ = performance
        wrong_cases = []
        for i, d in enumerate(self.dev_set):
            if perf_vec[i] < 0.5:
                wrong_cases.append(d[0])
                if len(wrong_cases) >= self.num_feedback_wrongcases:
                    break
        wrong_cases = "\n".join(wrong_cases)

        if self.examiner_message is None:
            message = AgentMessage(
                query="""You are a quick improver. Given an existing prompt and a series of cases where it
made mistakes. Look through each case carefully and identify what is causing the
mistakes. Based on these observations, output ways to improve the prompts based
on the mistakes.
## Existing Prompt ##
{context.candidate}
## Cases where it gets wrong: ##
{context.wrong_cases}
ways to improve the existing prompt based on observations of the mistakes in the
cases above are:
""",
                context={"candidate": candidate, "wrong_cases": wrong_cases},
            )
        else:
            message = self.examiner_message
            if message.context is None:
                message.context = {"candidate": candidate, "wrong_cases": wrong_cases}
            else:
                message.context["candidate"] = candidate
                message.context["wrong_cases"] = wrong_cases
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
        """
        Generate another prompt with the same semantic meaning and intentions.

        This method works with corresponding semantic_message and semantic_chain attributes.
        In the prompt in semantic_message, it must contain  key 'context.candidate' to parse with the candidate input.

        Args:
            candidate (str): the original prompt
            **kwargs: arguments for semantic chains

        Returns:
            str: the generated prompt
        """
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
                perf = self._scorer(
                    self.base_chain, prompt, P, S, self.dev_set, **self._scorer_args
                )
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)
        else:
            while len(P) < self.pool_size_0:
                prompt = self.lamarckian(self.init_data)
                perf = self._scorer(
                    self.base_chain, prompt, P, S, self.dev_set, **self._scorer_args
                )
                if perf is not None:
                    P.append(prompt)
                    S.append(perf)

        # phase 1: local feedback operation
        # use feedback to generate new prompt
        t = 0
        k = 0
        old_score = max([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_1:
            P_t = []
            S_t = []
            for prompt, s in zip(P, S):
                new_prompt = self.feedback(prompt, s)
                new_perf = self._scorer(
                    self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
                )
                if new_perf is not None:
                    P_t.append(new_prompt)
                    S_t.append(new_perf)

            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            P, S = SEEPromptAugmenter._sort_pool(
                [*P, *P_t], [*S, *S_t], self.pool_size_1
            )
            new_score = max([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        # phase 2: global fusion operation
        # use eda and crossover to fusion existing prompt and generate new prompt
        t = 0
        k = 0
        old_score = np.mean([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_2:
            new_prompt = self.eda(P) if k % 2 else self.crossover(P, S)
            new_perf = self._scorer(
                self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
            )
            if new_perf is not None:
                P.append(new_prompt)
                S.append(new_perf)

            # Performance gain is defined as whether the average performance of the pool is improved
            # Note: we calculate the average score using highest score members and limited by the pool size
            P, S = SEEPromptAugmenter._sort_pool(P, S, self.pool_size_2)
            new_score = np.mean([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        # phase 3: local semantic operation
        # use semantic to generate new prompts
        t = 0
        k = 0
        old_score = max([s[1] for s in S])
        while t < self.performance_gain_threshold and k <= self.tolerance_3:
            P_t = []
            S_t = []
            for prompt in P:
                new_prompt = self.semantic(prompt)
                new_perf = self._scorer(
                    self.base_chain, new_prompt, P, S, self.dev_set, **self._scorer_args
                )
                if new_perf is not None:
                    P_t.append(new_prompt)
                    S_t.append(new_perf)

            # Performance gain is defined as whether the new candidate has a higher performance than its parent
            P, S = SEEPromptAugmenter._sort_pool(
                [*P, *P_t], [*S, *S_t], self.pool_size_3
            )
            new_score = max([s[1] for s in S])
            t = (new_score - old_score) / old_score
            k += 1

        message.query = P[np.argmax([s[1] for s in S])]
        return message
