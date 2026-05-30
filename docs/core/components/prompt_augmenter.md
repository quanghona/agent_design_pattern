# Prompt Augmenter

## Overview

The `PromptAugmenter` component enhances or rewrites prompts before they are sent to an LLM. It sits between the input guardrail and the generation stage in a `TypicalLLMChain` pipeline, transforming `AgentMessage` objects by adding external context or restructuring the prompt text. There are two broad categories of prompt augmentation: **data augmentation**, which injects external information such as retrieved documents, structured data, or RAG results into the prompt; and **structural augmentation**, which rewrites or refines the prompt text itself, potentially using an LLM to generate a better-formulated version.

## Architecture

Prompt augmenters inherit from `BaseChain` and integrate directly into the chain pipeline. When a chain executes, the message flows through the input guardrail, then through the configured `BasePromptAugmenter`, then through the generation step, and finally through the output guardrail. The augmenter receives an `AgentMessage` and returns a modified `AgentMessage` with an updated `query` field.

The framework provides four concrete implementations:

- **`IdentityPromptAugmenter`** — a no-op default that passes messages through unchanged.
- **`SimplePromptAugmenter`** — concatenates the original prompt with external data using a configurable template.
- **`MetaPromptAugmenter`** — delegates prompt rewriting to an LLM chain.
- **`DeduplicationPromptAugmenter`** — removes exact and near-duplicate sentences from the prompt (documented below).
- **`SEEPromptAugmenter`** — performs strategic exploration and exploitation for in-context prompt optimization (documented below).
- **`RLPromptAugmenter`** — uses reinforcement learning to iteratively select and apply augmenters for prompt optimization (documented below).

![diagram](img/prompt_augmenter-architecture.jpg)

## Key Concepts

### Loop Control

All prompt augmenters support optional looping via the `loop` parameter on `BasePromptAugmenter`. This allows the same augmenter to be applied multiple times:

- **Integer**: Apply the augmenter a fixed number of times.
- **Callable**: Accept a function `AgentMessage -> bool` that returns `True` to continue looping and `False` to stop.
- **`None`** (default): Apply the augmenter exactly once.

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Apply augmentation 3 times
augmenter = SimplePromptAugmenter(
    format="{query}\n\nAdditional context:\n{data}",
    data_key="context.data",
    loop=3,
)

# Apply until the query length exceeds 500 characters
augmenter = SimplePromptAugmenter(
    format="{query}\n\n{data}",
    data_key="context.data",
    loop=lambda msg: len(msg.query) < 500,
)
```

### Data Augmentation with SimplePromptAugmenter

`SimplePromptAugmenter` is the simplest data augmentation strategy. It takes the original prompt text and concatenates it with external data stored in the `AgentMessage.context` dictionary, using a user-defined format string. The format string must contain both `{query}` (the original prompt) and `{data}` (the external data). Additional keyword arguments passed to `augment()` are also interpolated into the format string.

The `data_key` parameter specifies where to find the data in `message.context`. It must start with the prefix `context.` — for example, `context.data` reads from `message.context["data"]`. This convention allows the same format string to reference multiple data sources if needed.

### Structural Augmentation with MetaPromptAugmenter

`MetaPromptAugmenter` delegates prompt rewriting to an LLM chain. Instead of template-based concatenation, it passes the `AgentMessage` through a `BaseLLMChain` (such as a LangChain or LlamaIndex chain) that can reason about the prompt and produce a rewritten version. This is useful when the prompt needs semantic restructuring, tone adjustment, or intelligent context selection that a simple template cannot achieve.

### Deduplication with DeduplicationPromptAugmenter

`DeduplicationPromptAugmenter` removes exact and near-duplicate sentences from a prompt. Repeated sentences make it harder for an LLM to focus on the correct information and waste token budget. This augmenter tokenizes the prompt into sentences using NLTK, applies a deduplication algorithm, and reassembles the unique sentences.

The class supports six algorithms, each suited to different scenarios:

| Algorithm | Type | Best For |
|-----------|------|----------|
| `minhash` | Exact (via MinHash fingerprint) | Small to medium prompts; fast and accurate exact deduplication |
| `minhash_lsh` | Near-duplicate (via LSH indexing) | Large prompts where O(n²) comparison is too slow |
| `bloomfilter` | Exact (probabilistic) | Very fast exact deduplication with minimal memory |
| `simhash` | Near-duplicate (Hamming distance) | Detecting semantically similar sentences with minor word changes |
| `lsh_bloom` | Near-duplicate (space-efficient LSH) | Very large-scale deduplication with bounded memory |
| `suffix_array` | Exact substring | Finding duplicate character-level substrings across sentences |

The class does **not** use semantic deduplication. Semantic similarity helps strengthen context and can improve model performance, so near-duplicate sentences that carry distinct information are preserved unless the algorithm explicitly detects them as near-duplicates (e.g., simhash, minhash_lsh).

#### Algorithm: MinHash (Exact)

Uses C-MinHash with `CMinHashDeduplicator` for exact deduplication based on MinHash fingerprint comparison. Each sentence is tokenized into word tokens, a MinHash signature is computed, and the deduplicator checks whether a similar fingerprint already exists. This is the default algorithm and works well for most use cases.

**When to use**: Small to medium-sized prompts where you want fast, accurate exact deduplication without configuring LSH parameters.

#### Algorithm: MinHash-LSH (Near-Duplicate)

Uses R-MinHash with Locality-Sensitive Hashing (LSH) for efficient near-duplicate detection at scale. Instead of comparing every sentence against every other sentence (O(n²)), LSH partitions sentences into buckets so that similar sentences land in the same bucket. This reduces comparison to approximately O(n). The `num_bands` and `num_perm` parameters control the sensitivity.

**When to use**: Large prompts with many sentences where O(n²) comparison becomes a bottleneck. Ideal when you want to catch near-duplicates (slightly reworded sentences) rather than just exact matches.

#### Algorithm: Bloom Filter (Exact)

Uses the `rbloom` library's Bloom Filter for fast probabilistic exact deduplication. A Bloom Filter is a space-efficient probabilistic data structure that can tell you whether an element is definitely not in a set or probably in a set. Sentences are compared as whole strings (normalized by case). This is the fastest algorithm with the smallest memory footprint.

**When to use**: When you need the fastest possible exact deduplication and can tolerate a small false positive rate (configurable via `false_positive_rate`). Best for simple exact-match scenarios.

#### Algorithm: SimHash (Near-Duplicate)

Uses a custom numpy-based SimHash implementation that computes a fingerprint for each sentence and compares fingerprints using Hamming distance. Sentences with similarity above the `threshold` are considered near-duplicates. Unlike MinHash, SimHash operates on character-level token weights, making it more sensitive to minor word substitutions.

**When to use**: When you want to detect sentences that are semantically similar with minor word changes (e.g., "The model achieved 95% accuracy" vs. "The model reached 95 percent accuracy"). Good for catching paraphrased content.

#### Algorithm: LSH-Bloom (Near-Duplicate, Space-Efficient)

Combines MinHash with an LSH-Bloom Filter for space-efficient near-duplicate detection at very large scale. Inspired by the algorithm from [arXiv:2411.04257](https://arxiv.org/abs/2411.04257), it uses MinHash signatures as keys into a Bloom Filter organized by LSH buckets. This provides the near-duplicate detection of LSH with the memory efficiency of Bloom Filters.

**When to use**: Very large-scale deduplication where memory is constrained but you still need near-duplicate detection. Requires configuring both LSH parameters (`num_bands`, `num_perm`) and Bloom Filter parameters (`expected_items`, `false_positive_rate`).

#### Algorithm: Suffix Array (Exact Substring)

Uses a Suffix Array with Longest Common Prefix (LCP) array for exact substring deduplication at the character level. Inspired by Google's `deduplicate-text-datasets` and Chenghao Mou's `text-dedup`, this algorithm identifies duplicate character sequences across all sentences, then removes sentences whose entire content falls within duplicate regions. It keeps the first occurrence of each duplicate sentence.

**When to use**: When you need to find exact character-level duplicate substrings, including partial overlaps between sentences. Useful for detecting copy-paste duplication that spans sentence boundaries.

### Strategic Prompt Optimization with SEEPromptAugmenter

`SEEPromptAugmenter` implements **Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization** ([arXiv:2402.11347](https://arxiv.org/abs/2402.11347)). It uses LLM-powered operators to iteratively generate, evaluate, and refine prompt candidates through a four-phase optimization loop, ultimately selecting the prompt with the highest performance on a development set.

The framework maintains a **prompt pool** (candidate prompts) and a **performance pool** (corresponding evaluation scores). Through successive phases of exploration (generating diverse candidates) and exploitation (refining high-performing candidates), it converges on an optimized prompt.

> **Experimental use**: This component requires many LLM API calls and is intended for experimental/prompt-engineering use, not production. A `UserWarning` is emitted on instantiation.

#### Five LLM Operators

SEE defines five operators that transform prompt candidates:

| Operator | Role | Phase(s) | Description |
|----------|------|----------|-------------|
| **Lamarckian** | Exploration | Phase 0 | Reverse-engineers a prompt from a set of input-output pairs. Given a dataset, the LLM generates a prompt that would elicit the correct outputs. |
| **EDA** (Estimation of Distribution) | Exploration | Phase 2 | Studies a group of high-performing candidate prompts and generates a new candidate by learning from their common patterns. |
| **Crossover** | Exploration | Phase 2 | Mixes traits from multiple parent prompts to generate a new candidate. Generalized from the original paper's 2-parent limit to support any number of parents. |
| **Feedback** | Exploitation | Phase 1 | Uses a two-agent pipeline (Examiner + Improver) to analyze a candidate's failures and generate an improved version. The Examiner identifies why the prompt fails on certain cases; the Improver generates a fix. |
| **Semantic** | Exploitation | Phase 3 | Lexically modifies a prompt while preserving its semantic meaning. Generates paraphrased variants that may perform differently due to LLM sensitivity to wording. |

#### Four Optimization Phases

The `augment()` method runs through four sequential phases, each with its own pool size, tolerance, and performance gain threshold:

```
Phase 0: Global Initialization          → Lamarckian or Semantic
Phase 1: Local Feedback Operation       → Feedback (Examiner + Improver)
Phase 2: Global Fusion Operation        → EDA + Crossover (alternating)
Phase 3: Local Semantic Operation       → Semantic
```

- **Phase 0** (pool size: 15): Initializes the prompt pool. If `init_data` is a dataset (list of input-output pairs), uses Lamarckian to generate prompts from the data. If `init_data` is a string (an existing prompt), uses Semantic to generate diverse variants.
- **Phase 1** (pool size: 5): Applies Feedback to each candidate. The Examiner identifies wrong cases from the development set; the Improver generates fixes. Performance gain is measured as improvement over the best parent.
- **Phase 2** (pool size: 5): Applies EDA and Crossover alternately. EDA studies the distribution of top candidates; Crossover combines parent traits. Performance gain is measured as improvement in average pool score.
- **Phase 3** (pool size: 5): Applies Semantic mutation to each candidate, generating paraphrased variants. Performance gain is measured as improvement over the best parent.

Each phase terminates early if either:
- The performance gain drops below `performance_gain_threshold` (default: 1%), or
- The tolerance counter (`K_1`, `K_2`, `K_3`) is exceeded without improvement.

#### Extensions Beyond the Original Paper

Our implementation extends the original SEE framework in several ways:

1. **Generalized Crossover (CR+D)**: The original paper's Crossover operator accepts exactly 2 parents. We generalize it to accept any number `k` of parents. When `crossover_with_distinct=True`, we select parents that maximize the total pairwise distance between their performance vectors (Crossover + Distinct, or CR+D). This is implemented via a greedy algorithm in `max_vector_distance_subarray()` that iteratively selects the point adding the most extra distance to the current set.

2. **Multiple Parent Selection Strategies**: EDA and Crossover support three parent selection strategies:
   - **Random**: Uniform random selection without replacement.
   - **Wheel**: Fitness-proportionate selection (higher-performing candidates are more likely to be selected).
   - **Tournament**: Random pairwise tournaments; the winner of each pair is selected.

3. **Configurable EDA Ranking**: The `eda_with_index` flag controls whether EDA preserves the ranking of selected candidates (`True`) or shuffles them (`False`) before passing to the LLM.

4. **Custom Scorer**: Beyond the default Hamming scorer, users can provide a custom scorer function and distance function. The scorer signature is `Callable[[BaseLLMChain, str, Sequence[str], Sequence[SEEPerformanceTuple], SEEDataSet, ...], SEEPerformanceTuple | None]`. The scorer returns `None` to reject a candidate (e.g., if it's too similar to existing pool members).

5. **Custom Evaluation Method**: The `eval_method` parameter controls how the LLM's output is compared against the expected answer. Options include `"exact"` (string equality), `"include"` (substring check), or a custom callable. This enables evaluation methods like BLEU, ROUGE, BERTScore, or LLM-as-a-judge.

6. **Default Prompt Templates**: Each operator has a built-in default prompt template loaded from `aap_core.default_prompts` package resources. Users can override these by providing custom `AgentMessage` objects via the `*_message` fields.

7. **Framework Integration**: The implementation integrates with the `aap_core` framework, using `AgentMessage` for data passing (with `context` dictionary for operator inputs) and `BaseLLMChain` for LLM calls. This allows seamless composition with other framework components.

8. **Flexible LLM Assignment**: Each operator can use a different LLM chain. Users can assign different models (e.g., a strong model for Crossover, a fast model for Semantic) to different operators.

#### Temperature Guidelines

To maximize operator effectiveness, adjust the LLM temperature per phase:
- **Higher temperature** for exploration phases (0 and 2) and global operators (Lamarckian, EDA, Crossover) — encourages diversity.
- **Lower temperature** for exploitation phases (1 and 3) and local operators (Feedback, Semantic) — encourages focused refinement.

Temperature adjustment is handled at the chain level, outside this class.

### Reinforcement Learning with RLPromptAugmenter

`RLPromptAugmenter` is the most sophisticated prompt augmenter in the framework. It formulates prompt optimization as a **reinforcement learning (RL) problem**, where an intelligent agent learns to iteratively select and apply prompt augmenters to maximize a reward signal. Unlike `SEEPromptAugmenter`, which uses LLM-powered operators to rewrite prompts directly, `RLPromptAugmenter` learns a **policy** — a probability distribution over available augmenters — that determines which augmenter to apply at each step.

This approach is particularly powerful when:
- You have a diverse set of augmenters (deduplication, formatting, context injection, etc.) and want the system to learn which combination works best.
- The reward signal is well-defined (e.g., a quality scorer, task accuracy, or human preference).
- You want to automate prompt engineering without manual LLM-based optimization.

> **Note**: `RLPromptAugmenter` requires a trained policy model. See [Policy Trainer](policy_trainer.md) for training instructions.

#### RL Problem Formulation

The prompt optimization problem is cast as a **Markov Decision Process (MDP)** with the following components:

| RL Component | Description | Implementation |
|--------------|-------------|----------------|
| **State / Observation** | The current prompt, represented as an embedding vector | `PromptOptimizationEnv` computes the embedding via a user-provided `embedding_model` |
| **Action** | Selecting one of the available prompt augmenters to apply | `gymnasium.spaces.Discrete(num_augmenters)` — each action index maps to a `BasePromptAugmenter` |
| **Reward** | Quality score of the resulting prompt after applying the selected augmenter | Computed by a user-provided `reward_model` callable, minus a duplication penalty |
| **Transition** | Applying an augmenter modifies the current prompt, leading to a new state (new prompt embedding) | `PromptOptimizationEnv.step()` applies the selected augmenter and returns the new embedding |
| **Episode Termination** | Episode ends when `max_steps` is reached, the reward exceeds a threshold, or the prompt embedding changes too little | Configurable via `reward_threshold` and `min_embedding_threshold` |

The RL formulation decomposes into two sub-systems:

**1. The Agent (Policy)** — The control algorithm that expresses how we act. The policy model takes the current prompt embedding as input and outputs action logits, which are converted into a probability distribution over augmenters. During training, the policy is updated via policy gradient methods (PPO, REINFORCE++, GRPO). During inference, the trained policy selects augmenters to iteratively optimize the prompt.

**2. The Environment** — The external factor that is not part of the agent itself. The agent only knows the environment through the state (observation), action, and reward signals. In our case, the environment is implemented as `PromptOptimizationEnv`, a `gymnasium` environment that:
- Maintains the current prompt text and its embedding.
- Provides a discrete action space (one action per available augmenter).
- Computes rewards using a user-defined reward model.
- Applies augmenters to transition between states.
- Detects episode termination conditions (max steps, reward threshold, embedding stagnation).

#### Workflow: Training vs. Inference

The `RLPromptAugmenter` workflow has two distinct phases:

**Training Phase** — The policy is improved through interaction with the environment:

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop                                          │
│                                                         │
│  for each episode:                                      │
│    1. Reset environment → initial prompt embedding      │
│    2. for each step:                                    │
│       a. Policy observes embedding → outputs action logits  │
│       b. Sample action (explore/exploit)                │
│       c. Environment applies selected augmenter         │
│       d. Reward model scores the new prompt             │
│       e. Policy gradient update (PPO / REINFORCE++ / GRPO) │
│    3. Check early stopping (no improvement)             │
│    4. Save checkpoint                                   │
└─────────────────────────────────────────────────────────┘
```

The training loop is implemented in `BasePolicyTrainer.fit()` and its subclasses (`PPOTrainer`, `ReinforcePPTrainer`, `GRPOTrainer`). Key training components:

- **Policy Model**: A neural network (e.g., `GPT2Policy`, `GPT2RoPEGQAPolicy`) that maps prompt embeddings to action logits.
- **Exploration Strategy**: `EpsilonGreedyExploration` with linear decay from `eps_init` to `eps_final`, or `RandomExploration` for fixed ratios.
- **Replay Buffer**: `SimpleReplayBuffer` for off-policy trainers, or no buffer for on-policy trainers.
- **Optimization**: Adam optimizer with optional learning rate scheduler.
- **Logging**: Weights & Biases (WandB) integration for tracking episode rewards, action loss, entropy, and KL divergence.
- **Checkpointing**: Periodic model saving for resuming training.

**Inference Phase** — The trained policy interacts with the environment to optimize a prompt:

```
┌─────────────────────────────────────────────────────────┐
│  Inference (RLPromptAugmenter.augment)                  │
│                                                         │
│  1. Extract initial prompt from AgentMessage.query      │
│  2. Reset environment with initial prompt               │
│  3. while step_count < max_steps:                       │
│       a. Policy observes embedding → selects action     │
│       b. Environment applies selected augmenter         │
│       c. If terminated or truncated → break             │
│  4. Return AgentMessage with optimized prompt           │
└─────────────────────────────────────────────────────────┘
```

During inference, the policy operates **deterministically** (no exploration) — it always selects the action with the highest probability.

#### Environment and Reward Design

The `PromptOptimizationEnv` class is the bridge between the RL framework and the prompt augmentation domain. It is a `gymnasium` environment that manages the prompt optimization lifecycle.

**Environment Configuration:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_prompt` | `str` | *(required)* | The starting prompt for the episode |
| `augmenters` | `Sequence[BasePromptAugmenter]` | *(required)* | List of augmenters available as actions. Each augmenter maps to one action index |
| `embedding_model` | `Callable[[str], np.ndarray]` | *(required)* | Converts prompt text to an embedding vector. Common choices: `sentence-transformers` models like `BAAI/bge-small-en-v1.5` |
| `reward_model` | `Callable[[str], float]` | *(required)* | Scores the quality of a prompt. Common choices: `agentlans/bge-small-en-v1.5-prompt-quality` or task-specific accuracy metrics |
| `max_steps` | `int` | `10` | Maximum number of steps per episode |
| `min_embedding_threshold` | `float` | `0.8` | Episode terminates if cosine similarity between consecutive prompt embeddings falls below this threshold (prevents meaningless changes) |
| `reward_threshold` | `float` | `inf` | Episode terminates if the reward exceeds this threshold (useful when a perfect score is achievable) |
| `penalty_weight` | `float` | `0.2` | Weight for the duplication penalty. Final reward = `base_reward - (duplication_penalty * penalty_weight)` |

**Reward Design:**

The reward signal is the most critical design choice in RL prompt optimization. The environment computes the reward as:

$$\text{reward} = \text{reward\_model}(\text{prompt}) - (\text{duplication\_penalty} \times \text{penalty\_weight})$$

The **duplication penalty** uses a Bloom filter to detect repeated sentences and penalizes the reward proportionally to $\sqrt{\text{total\_duplicate\_chars}}$. This prevents the policy from repeatedly applying augmenters that introduce redundant content.

**Common Reward Models:**

| Reward Model | Source | Use Case |
|-------------|--------|----------|
| `agentlans/bge-small-en-v1.5-prompt-quality` | Hugging Face | General prompt quality scoring |
| Task accuracy | Custom function | Maximize task-specific performance (e.g., classification accuracy on a dev set) |
| Length penalty | Custom function | Encourage concise prompts |
| Human preference | Custom function | Optimize toward human-rated quality |

**Policy Model Design:**

The policy model is a neural network that maps prompt embeddings to action probabilities. The framework provides two built-in models:

| Model | Architecture | Description |
|-------|-------------|-------------|
| `GPT2Policy` | GPT-2 (minGPT) | Transformer-based policy with causal self-attention. Observations are projected to the embedding dimension and processed through transformer blocks. Supports value head for critic-based methods. |
| `GPT2RoPEGQAPolicy` | Upgraded GPT-2 | Enhanced policy with RMSNorm, RoPE (Rotary Positional Embeddings), and Grouped Query Attention (GQA). Better performance with fewer parameters. |

Both models inherit from `BasePolicy` and implement:
- `forward(obs)`: Maps observation tensor to action logits.
- `get_action(logits, deterministic)`: Samples an action from the logits (deterministic argmax or stochastic sampling).
- `evaluate_actions(obs, actions, masks)`: Computes log probabilities, entropy, and value estimates for policy gradient updates.
- `save(path)` / `load(path)`: Checkpoint management.

**Trainer Design:**

The trainer classes inherit from `BasePolicyTrainer` and implement different policy gradient algorithms. Most trainers are inherited from RLHF (Reinforcement Learning from Human Feedback) tasks and adapted for prompt optimization:

| Trainer | Algorithm | On/Off-Policy | Critic | Description |
|---------|-----------|---------------|--------|-------------|
| `PPOTrainer` | Proximal Policy Optimization | On-policy | Optional | The most widely used policy gradient method. Uses clipped surrogate objective for stable updates. Supports KL loss for controlling policy drift. |
| `ReinforcePPTrainer` | REINFORCE++ | On-policy | None | A simplified approach using mean reward baseline and PPO-style clipping, but without a critic network. Uses cumulative returns directly. |
| `GRPOTrainer` | Group Relative Policy Optimization | On/Off-policy | None | Uses group-relative advantages (comparing outputs from the same prompt). Supports both on-policy and off-policy modes with replay buffer. |

**Replay Buffers** (for off-policy trainers):

| Buffer | Sampling Strategy | Description |
|--------|-------------------|-------------|
| `SimpleReplayBuffer` | Uniform random | FIFO circular buffer with uniform sampling. Suitable for off-policy trainers. |
| `PrioritizedReplayBuffer` | Advantage-weighted | Prioritizes transitions with higher TD-error (or advantage). Uses a binary heap for O(log N) priority updates. |

#### Basic Example: RLPromptAugmenter Inference

```python
from aap_core import AgentMessage
from aap_core.policy import GPT2RoPEGQAPolicy
from aap_core.prompt_augmenter import (
    IdentityPromptAugmenter,
    PromptOptimizationEnv,
    RLPromptAugmenter,
    SimplePromptAugmenter,
)
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. Define the embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 2. Define the reward model (prompt quality scorer)
reward_model = SentenceTransformer("agentlans/bge-small-en-v1.5-prompt-quality")

# 3. Define available augmenters (actions)
augmenters = [
    IdentityPromptAugmenter(),  # Action 0: do nothing
    SimplePromptAugmenter(
        format="{query}\n\nAdditional context:\n{data}",
        data_key="context.data",
    ),  # Action 1: add context
]

# 4. Create the environment
env = PromptOptimizationEnv(
    initial_prompt="Explain quantum entanglement.",
    augmenters=augmenters,
    embedding_model=embedding_model.encode,
    reward_model=lambda prompt: float(reward_model.encode(prompt)),
    max_steps=5,
)

# 5. Load the trained policy model
policy_model = GPT2RoPEGQAPolicy.load("./ckpt/best_policy.pt")

# 6. Create the RL Prompt Augmenter
rl_augmenter = RLPromptAugmenter(env=env, policy_model=policy_model)

# 7. Use it in a chain
message = AgentMessage(query="Explain quantum entanglement.")
result = rl_augmenter(message)
print(result.query)  # The optimized prompt
```

#### Advanced Example: Training an RLPromptAugmenter

```python
from aap_core.policy import GPT2RoPEGQAPolicy
from aap_core.policy_trainer import EpsilonGreedyExploration, PPOTrainer
from aap_core.prompt_augmenter import (
    IdentityPromptAugmenter,
    PromptOptimizationEnv,
    SimplePromptAugmenter,
)
from sentence_transformers import SentenceTransformer
from gymnasium import spaces
import torch

# Load models
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reward_model = SentenceTransformer("agentlans/bge-small-en-v1.5-prompt-quality")

# Define augmenters
augmenters = [
    IdentityPromptAugmenter(),
    SimplePromptAugmenter(
        format="{query}\n\nContext:\n{data}",
        data_key="context.data",
    ),
]

# Create environment
env = PromptOptimizationEnv(
    initial_prompt="Explain AI.",
    augmenters=augmenters,
    embedding_model=embedding_model.encode,
    reward_model=lambda prompt: float(reward_model.encode(prompt)),
    max_steps=5,
)

# Create policy model
policy_model = GPT2RoPEGQAPolicy(
    action_space=env.action_space,
    observation_space=env.observation_space,
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=64,
)

# Set up optimizer and scheduler
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=1e-6
)

# Set up exploration
exploration = EpsilonGreedyExploration(
    eps_init=0.5,
    eps_final=0.05,
    decay_episodes=1000,
)

# Create trainer
trainer = PPOTrainer(
    policy_model=policy_model,
    env=env,
    max_episodes=1000,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    exploration_module=exploration,
    clip_param=0.2,
    num_mini_batch=192,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=1.0,
    gamma=0.99,
)

# Train
trainer.fit(
    checkpoint_every=100,
    earlystop_last=600,
    record_every=10,
    use_wandb=True,
    wandb_project="prompt_optimization",
    checkpoint_dir="./ckpt",
)

# Save the best policy
policy_model.save("./ckpt/best_policy.pt")
```

#### Configuration

`RLPromptAugmenter` has minimal configuration — it delegates most settings to the environment and the trained policy:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | `PromptOptimizationEnv` | *(required)* | The prompt optimization environment containing the augmenters, embedding model, and reward model |
| `policy_model` | `BasePolicy` | *(required)* | The trained policy model that selects augmenters |

The environment (`PromptOptimizationEnv`) and policy model (`BasePolicy` subclasses) have their own configuration options documented in their respective sections above.

#### When to Use RLPromptAugmenter

`RLPromptAugmenter` is the right choice when:

- **You have a well-defined reward signal**: The reward model must provide meaningful, differentiable feedback. A good reward model is the single most important factor for success.
- **You want to automate augmenter selection**: Instead of manually composing augmenters, let the policy learn the optimal sequence.
- **You have diverse augmenters**: The more augmenters in the action space, the more the policy can learn to combine them effectively.
- **You can afford training time**: Training typically requires hundreds to thousands of episodes. Use early stopping to halt when performance plateaus.

`RLPromptAugmenter` is **not** the right choice when:

- **You need fast, deterministic augmentation**: The RL inference loop is slower than a single augmenter application.
- **You lack a good reward model**: Poor rewards lead to poor policies. If you cannot define a meaningful reward, use `SEEPromptAugmenter` instead.
- **You have only one or two augmenters**: The RL approach shines with diverse action spaces. For simple cases, `SimplePromptAugmenter` or `MetaPromptAugmenter` is sufficient.

#### See Also

- [Policy Trainer](policy_trainer.md) — Training policy models for `RLPromptAugmenter`
- [PromptOptimizationEnv](#reinforcement-learning-with-rlpromptaugmenter) — The RL environment for prompt optimization
- [SEEPromptAugmenter](#strategic-prompt-optimization-with-seepromptaugmenter) — LLM-based prompt optimization (alternative to RL)
- [RL Prompt Augmenter Training Example](../../example/transformers/rl_prompt_augmenter_training.ipynb) — Complete training notebook

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import IdentityPromptAugmenter

# Identity augmenters pass messages through unchanged
augmenter = IdentityPromptAugmenter()

message = AgentMessage(query="What is AI?")
result = augmenter(message)
print(result.query)  # "What is AI?"
```

### Basic Example: Simple Data Augmentation

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Define the format: original query + external data
augmenter = SimplePromptAugmenter(
    format="Question: {query}\n\nContext:\n{data}",
    data_key="context.retrieved_docs",
)

message = AgentMessage(
    query="What is reinforcement learning?",
    context={
        "retrieved_docs": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards."
    },
)
result = augmenter(message)
print(result.query)
# "Question: What is reinforcement learning?\n\nContext:\nReinforcement learning is a type of machine learning..."
```

### Advanced Example: Simple Augmenter with Multiple Data Sources

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SimplePromptAugmenter

# Format string can include additional kwargs beyond query and data
augmenter = SimplePromptAugmenter(
    format="You are an expert. Answer the question using the provided context.\n\nQuestion: {query}\n\nContext:\n{data}\n\nConstraints: {constraints}",
    data_key="context.knowledge_base",
)

message = AgentMessage(
    query="Explain quantum entanglement.",
    context={
        "knowledge_base": "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated in such a way that the quantum state of each particle cannot be described independently.",
    },
)
result = augmenter(
    message,
    constraints="Keep the explanation under 200 words and avoid mathematical notation.",
)
print(result.query)
# "You are an expert. Answer the question using the provided context.
#
# Question: Explain quantum entanglement.
#
# Context:
# Quantum entanglement is a physical phenomenon...
#
# Constraints: Keep the explanation under 200 words..."
```

### Advanced Example: Meta Prompt Augmenter with an LLM Chain

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import MetaPromptAugmenter

# Assume you have a LangChain or LlamaIndex chain configured
# with a prompt template that instructs the LLM to rewrite the prompt
from aap_langchain import LangChainLLMChain

llm_chain = LangChainLLMChain(
    # ... configure your LLM and prompt template ...
    name="prompt-rewriter",
)

augmenter = MetaPromptAugmenter(chain=llm_chain)

message = AgentMessage(query="tell me about ai")
result = augmenter(message)
# The LLM chain rewrites the query to be more structured and detailed
print(result.query)
# "Please provide a comprehensive explanation of artificial intelligence, including its key subfields..."
```

### Basic Example: Deduplication Prompt Augmenter (MinHash)

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter

# Remove duplicate sentences using MinHash
augmenter = DeduplicationPromptAugmenter(
    algo_args={
        "algorithm_name": "minhash",
        "threshold": 0.9,
        "num_perm": 128,
        "seed": 42,
    },
)

message = AgentMessage(
    query="AI is a field of computer science. AI focuses on building smart machines. "
          "AI is a field of computer science. These machines can perform tasks that "
          "typically require human intelligence.",
)
result = augmenter(message)
print(result.query)
# "AI is a field of computer science. AI focuses on building smart machines. "
# "These machines can perform tasks that typically require human intelligence."
```

### Basic Example: Bloom Filter Deduplication

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter

# Fast exact deduplication with Bloom Filter
augmenter = DeduplicationPromptAugmenter(
    algo_args={
        "algorithm_name": "bloomfilter",
        "expected_items": 1000,
        "false_positive_rate": 0.01,
    },
)

message = AgentMessage(
    query="The capital of France is Paris. Paris is known for the Eiffel Tower. "
          "The capital of France is Paris. The city has a rich history.",
)
result = augmenter(message)
print(result.query)
# "The capital of France is Paris. Paris is known for the Eiffel Tower. "
# "The city has a rich history."
```

### Advanced Example: MinHash-LSH for Large Prompts

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter

# Near-duplicate detection for large prompts with many sentences
# LSH provides O(n) complexity instead of O(n²)
augmenter = DeduplicationPromptAugmenter(
    algo_args={
        "algorithm_name": "minhash_lsh",
        "threshold": 0.8,
        "num_perm": 256,
        "num_bands": 16,
    },
)

message = AgentMessage(
    query=" ".join([
        "The model was trained on 1 million samples. "
        "The model was trained on one million samples. "  # near-duplicate
        "Accuracy reached 95 percent. "
        "Accuracy reached 95%. "  # near-duplicate
        "The dataset contains images from 50 categories. "
        "Training took 48 hours on a GPU cluster. "
    ] * 10),  # repeat to simulate a large prompt
)
result = augmenter(message)
# Near-duplicate sentences are removed, reducing token count
```

### Advanced Example: SimHash for Paraphrase Detection

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter

# Detect near-duplicates with minor word changes
augmenter = DeduplicationPromptAugmenter(
    algo_args={
        "algorithm_name": "simhash",
        "hash_bits": 64,
        "threshold": 0.85,
    },
)

message = AgentMessage(
    query="The neural network achieved state-of-the-art results on the benchmark. "
          "The neural network achieved state of the art results on the benchmark. "
          "This performance was verified across multiple test sets.",
)
result = augmenter(message)
# The paraphrased sentence is detected as a near-duplicate and removed
```

### Advanced Example: Suffix Array for Character-Level Duplication

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import DeduplicationPromptAugmenter

# Find exact duplicate substrings at the character level
augmenter = DeduplicationPromptAugmenter(
    algo_args={
        "algorithm_name": "suffix_array",
        "min_length": 20,
        "max_length": 10000,
    },
)

message = AgentMessage(
    query="Section 1: Introduction. This section introduces the problem. "
          "Section 1: Introduction. This section introduces the problem. "
          "Section 2: Methods. We describe our approach.",
)
result = augmenter(message)
# Duplicate sentences at the character level are removed
```

### Basic Example: SEEPromptAugmenter with Hamming Scorer

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SEEPromptAugmenter

# Define a development set for evaluation
# Each entry is a (question, expected_answer) pair
dev_set = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Hamlet?", "Shakespeare"),
]

# Define LLM chains for each operator
# Each chain can use a different model
from aap_langchain import LangChainLLMChain

base_chain = LangChainLLMChain(name="base", model="gpt-4")
lamarckian_chain = LangChainLLMChain(name="lamarckian", model="gpt-4")
eda_chain = LangChainLLMChain(name="eda", model="gpt-4")
crossover_chain = LangChainLLMChain(name="crossover", model="gpt-4")
examiner_chain = LangChainLLMChain(name="examiner", model="gpt-4")
improver_chain = LangChainLLMChain(name="improver", model="gpt-4")
semantic_chain = LangChainLLMChain(name="semantic", model="gpt-3.5")

# Initialize SEE with the default Hamming scorer
see = SEEPromptAugmenter(
    scorer="hamming",
    dist_func=SEEPromptAugmenter._hamming_distance,
    base_chain=base_chain,
    lamarckian_chain=lamarckian_chain,
    eda_chain=eda_chain,
    crossover_chain=crossover_chain,
    examiner_chain=examiner_chain,
    improver_chain=improver_chain,
    semantic_chain=semantic_chain,
    dev_set=dev_set,
    init_data=dev_set,  # Use Lamarckian for initialization
    pool_size_0=5,  # Smaller pool for faster experimentation
)

# Run the optimization
message = AgentMessage(query="Initial prompt here")
result = see(message)
print(result.query)  # The optimized prompt
```

### Advanced Example: SEEPromptAugmenter with Custom Scorer and Crossover + Distinct

```python
from aap_core import AgentMessage
from aap_core.prompt_augmenter import SEEPromptAugmenter
from aap_core.prompt_augmenter import SEEDataSet, SEEPerformanceTuple
from collections.abc import Sequence

# Custom scorer using exact match with a distance threshold
def custom_scorer(
    chain, prompt: str, pool: Sequence[str],
    performance_pool: Sequence[SEEPerformanceTuple],
    dataset: SEEDataSet, distance_threshold: int = 2,
) -> SEEPerformanceTuple | None:
    perf_vec, score = see.score(prompt, dataset)
    if not performance_pool:
        return (perf_vec, score)
    # Reject if too similar to any existing candidate
    min_dist = min(
        SEEPromptAugmenter._hamming_distance(perf_vec, p[0])
        for p in performance_pool
    )
    return None if min_dist < distance_threshold else (perf_vec, score)

see = SEEPromptAugmenter(
    scorer=custom_scorer,
    dist_func=SEEPromptAugmenter._hamming_distance,
    base_chain=base_chain,
    lamarckian_chain=lamarckian_chain,
    eda_chain=eda_chain,
    crossover_chain=crossover_chain,
    examiner_chain=examiner_chain,
    improver_chain=improver_chain,
    semantic_chain=semantic_chain,
    dev_set=dev_set,
    init_data=dev_set,
    # Enable Crossover + Distinct for diverse parent selection
    crossover_with_distinct=True,
    crossover_parent_selection="tournament",
    num_crossover_parents=3,  # Use 3 parents instead of 2
    # Use tournament selection for EDA
    eda_parent_selection="tournament",
    # Allow all candidates as EDA parents
    num_eda_parents=-1,
)

message = AgentMessage(query="Initial prompt")
result = see(message)
```

## Configuration

### BasePromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loop` | `int \| Callable[[AgentMessage], bool] \| None` | `None` | Loop control: an integer for fixed iterations, a callable for conditional looping, or `None` for single application |

### IdentityPromptAugmenter

No additional parameters. Inherits `loop` from `BasePromptAugmenter`.

### SimplePromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | *(required)* | Format string for the augmented prompt. Must contain `{query}` and `{data}`. Additional placeholders are filled by kwargs passed to `augment()`. |
| `data_key` | `str` | `"context.data"` | Key path in `message.context` to read the external data. Must start with `context.`. For example, `context.retrieved_docs` reads from `message.context["retrieved_docs"]`. |

**Validators:**
- `format` must contain both `{query}` and `{data}` placeholders.
- `data_key` must start with the prefix `context.`.

### MetaPromptAugmenter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain` | `BaseLLMChain` | *(required)* | An LLM chain that rewrites the prompt. The chain's `invoke()` method is called with the `AgentMessage` and returns the modified message. |

### DeduplicationPromptAugmenter

This class accepts a single `algo_args` parameter that configures the deduplication algorithm. The `algorithm_name` field selects which algorithm to use, and additional fields are passed to the algorithm-specific configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algo_args` | `Dict[str, Any]` | *(required)* | Dictionary containing algorithm configuration. Must include `algorithm_name` key. |

#### Algorithm-Specific `algo_args` Parameters

**minhash** (default)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"minhash"` | Must be `"minhash"` |
| `threshold` | `float` | `0.9` | MinHash similarity threshold for considering sentences as duplicates |
| `num_perm` | `int` | `128` | Number of permutations for MinHash signature computation |
| `seed` | `int` | `42` | Random seed for reproducibility |

**minhash_lsh**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"minhash_lsh"` | Must be `"minhash_lsh"` |
| `threshold` | `float` | `0.8` | Jaccard similarity threshold for near-duplicate detection |
| `num_perm` | `int` | `128` | Number of permutations for R-MinHash signature |
| `num_bands` | `int` | `16` | Number of LSH bands; fewer bands = more sensitive (more candidates) |

**bloomfilter**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"bloomfilter"` | Must be `"bloomfilter"` |
| `expected_items` | `int` | `1000` | Expected number of sentences; affects Bloom Filter size |
| `false_positive_rate` | `float` | `0.01` | Target false positive rate; lower = larger filter |

**simhash**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"simhash"` | Must be `"simhash"` |
| `hash_bits` | `int` | `64` | Number of bits in the SimHash fingerprint |
| `threshold` | `float` | `0.85` | Similarity threshold (based on Hamming distance); sentences with similarity >= threshold are duplicates |

**lsh_bloom**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"lsh_bloom"` | Must be `"lsh_bloom"` |
| `threshold` | `float` | `0.8` | Jaccard similarity threshold |
| `num_perm` | `int` | `128` | Number of permutations for MinHash |
| `num_bands` | `int` | `16` | Number of LSH bands |
| `expected_items` | `int` | `10000` | Expected number of items for Bloom Filter sizing |
| `false_positive_rate` | `float` | `0.01` | Target false positive rate |

**suffix_array**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm_name` | `str` | `"suffix_array"` | Must be `"suffix_array"` |
| `min_length` | `int` | `20` | Minimum substring length to consider as a duplicate |
| `max_length` | `int` | `10000` | Maximum substring length |

### SEEPromptAugmenter

This is the most complex prompt augmenter, implementing a full optimization loop. It requires multiple LLM chains (one per operator) and a development set for evaluation.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scorer` | `Literal["hamming"] \| Callable` | `"hamming"` | Scoring function. Use `"hamming"` for the default Hamming scorer, or provide a custom callable with signature `Callable[[BaseLLMChain, str, Sequence[str], Sequence[SEEPerformanceTuple], SEEDataSet, ...], SEEPerformanceTuple \| None]` |
| `dist_func` | `Callable[[Sequence[float], Sequence[float]], float] \| None` | `None` | Distance function for the scorer. Required when using a custom scorer. Use `SEEPromptAugmenter._hamming_distance` for Hamming distance. |
| `scorer_args` | `Dict` | `{}` | Additional arguments passed to the scorer function. |
| `eval_method` | `Literal["exact", "include"] \| Callable[[str, str], bool]` | `"exact"` | How to compare the LLM's output against the expected answer. `"exact"` for string equality, `"include"` for substring check, or a custom callable. |

#### Core Fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_chain` | `BaseLLMChain` | *(required)* | The base LLM chain used to generate results and evaluate prompts on the development set. |
| `lamarckian_chain` | `BaseLLMChain` | *(required)* | LLM chain for the Lamarckian operator (Phase 0 initialization from input-output pairs). |
| `eda_chain` | `BaseLLMChain` | *(required)* | LLM chain for the EDA operator (Phase 2 global fusion). |
| `crossover_chain` | `BaseLLMChain` | *(required)* | LLM chain for the Crossover operator (Phase 2 global fusion). |
| `examiner_chain` | `BaseLLMChain` | *(required)* | LLM chain for the Examiner agent in the Feedback operator (Phase 1). |
| `improver_chain` | `BaseLLMChain` | *(required)* | LLM chain for the Improver agent in the Feedback operator (Phase 1). |
| `semantic_chain` | `BaseLLMChain` | *(required)* | LLM chain for the Semantic operator (Phase 3). |
| `dev_set` | `SEEDataSet` | *(required)* | The development dataset for evaluating prompts. A sequence of `(input, expected_output)` tuples. Must have at least 1 entry. |
| `init_data` | `SEEDataSet \| str` | *(required)* | Initial data for Phase 0. If a dataset (list of pairs), uses Lamarckian to generate prompts. If a string, uses Semantic to mutate an existing prompt. |

#### Operator Message Fields (Optional)

Each operator has an optional `*_message` field that allows customizing the prompt template. If `None`, a default template is loaded from `aap_core.default_prompts`:

| Parameter | Default | Context Keys | Description |
|-----------|---------|--------------|-------------|
| `lamarckian_message` | `None` | `{context.pairs}` | Custom prompt for Lamarckian operator. |
| `eda_message` | `None` | `{context.candidates}` | Custom prompt for EDA operator. |
| `crossover_message` | `None` | `{context.parents}` | Custom prompt for Crossover operator. |
| `examiner_message` | `None` | `{context.candidate}`, `{context.wrong_cases}` | Custom prompt for the Examiner agent. |
| `improver_message` | `None` | `{context.candidate}`, `{context.feedback}` | Custom prompt for the Improver agent. |
| `semantic_message` | `None` | `{context.candidate}` | Custom prompt for Semantic operator. |

#### Phase Configuration Parameters

**Phase 0: Global Initialization**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size_0` | `int` | `15` | Maximum pool size for Phase 0 (marked as $n_0$ in the paper). |

**Phase 1: Local Feedback**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size_1` | `int` | `5` | Maximum pool size for Phase 1 (marked as $n_1$). |
| `tolerance_1` | `int` | `1` | Maximum iterations without improvement for Phase 1 (marked as $K_1$). |
| `num_feedback_wrongcases` | `int` | `3` | Number of wrong cases to include in the Examiner's prompt. Must be $> 0$. |

**Phase 2: Global Fusion**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size_2` | `int` | `5` | Maximum pool size for Phase 2 (marked as $n_2$). |
| `tolerance_2` | `int` | `8` | Maximum iterations without improvement for Phase 2 (marked as $K_2$). Applies to both EDA and Crossover combined. |
| `num_crossover_parents` | `int` | `2` | Number of parents for Crossover. `-1` uses all candidates. Generalized from the original paper's fixed 2 parents. |
| `num_eda_parents` | `int` | `-1` | Number of parents for EDA. `-1` uses all candidates. |
| `eda_with_index` | `bool` | `False` | Whether to preserve the ranking of EDA-selected candidates. |
| `crossover_with_distinct` | `bool` | `False` | Whether to select Crossover parents that maximize diversity (CR+D). |
| `eda_parent_selection` | `Literal["wheel", "random", "tournament"]` | `"random"` | Parent selection strategy for EDA. |
| `crossover_parent_selection` | `Literal["wheel", "random", "tournament"]` | `"random"` | Parent selection strategy for Crossover when `crossover_with_distinct=True`. |

**Phase 3: Local Semantic**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size_3` | `int` | `5` | Maximum pool size for Phase 3 (marked as $n_3$). |
| `tolerance_3` | `int` | `1` | Maximum iterations without improvement for Phase 3 (marked as $K_3$). |

**Global Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `performance_gain_threshold` | `float` | `0.01` | Performance gain threshold in percent scale. Phases terminate when gain drops below this value. |

## API Reference

See the full API reference: [`BasePromptAugmenter`][aap_core.prompt_augmenter.BasePromptAugmenter]

::: aap_core.prompt_augmenter.BasePromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.IdentityPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.SimplePromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.MetaPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.DeduplicationPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.SEEPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.RLPromptAugmenter
    options:
        show_root_heading: true
        show_signature: true

::: aap_core.prompt_augmenter.PromptOptimizationEnv
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Chain](chain.md) — How prompt augmenters integrate into the `TypicalLLMChain` pipeline
- [Retriever](retriever.md) — How retrieved data is stored in `AgentMessage.context` for augmentation
- [Dedup](dedup.md) — The underlying deduplication algorithms (Bloom, MinHash, SimHash, Suffix Array) used by `DeduplicationPromptAugmenter`
- [Policy Trainer](policy_trainer.md) — How to train a `BasePolicy` for use with `RLPromptAugmenter`
- [RL Prompt Augmenter Training Example](../../example/transformers/rl_prompt_augmenter_training.ipynb) — Complete training notebook for `RLPromptAugmenter`
