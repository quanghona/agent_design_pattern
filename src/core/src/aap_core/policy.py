"""Base policy classes for reinforcement learning.

This module provides abstract and concrete policy model classes that define
the interface for policy-based reinforcement learning algorithms. The policies
are designed to work with the PromptOptimizationEnv and similar gymnasium-based
environments.

Classes:
    BasePolicy: Abstract base class for all policy models
    GPT2Policy: GPT-2 based policy model using minGPT architecture
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


class BasePolicy(nn.Module, ABC):
    """Abstract base class for policy models.

    This class defines the interface for all policy models used in the
    reinforcement learning framework. Subclasses must implement the
    forward and evaluate_actions methods.

    The policy model takes observations as input and outputs action logits,
    which are then used to compute action probabilities and log probabilities
    for the taken actions.

    Args:
        action_space: The action space (typically gymnasium.spaces.Discrete)
        observation_space: The observation space (typically gymnasium.spaces.Box)

    Example:
        >>> from gymnasium import spaces
        >>> action_space = spaces.Discrete(5)
        >>> observation_space = spaces.Box(low=-1, high=1, shape=(768,))
        >>> policy = MyCustomPolicy(action_space, observation_space)
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        **kargs,
    ):
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy.

        Args:
            obs: Observations of shape (batch_size, obs_dim)

        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        pass

    def get_action(
        self, logits: torch.Tensor, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Tuple of (action, action_log_prob)
        """Extract action from policy output logits.

        This method takes the raw action logits and samples an action according
        to the specified mode (deterministic or stochastic). It also computes
        the log probability of the selected action.

        Args:
            logits: Action logits of shape (batch_size, action_dim)
            deterministic: Whether to select the most likely action (True) or sample from the distribution (False)
        Returns:
            Tuple of (action, action_log_prob)
                - action: Selected action of shape (batch_size,)
                - action_log_prob: Log probability of the selected action of shape (batch_size,)
        """
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions taken under the policy.

        This method computes the log probabilities of the taken actions,
        the entropy of the action distribution, and optionally value estimates.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            actions: Actions taken of shape (batch_size,)
            masks: Binary masks of shape (batch_size,) indicating
                   which positions are valid

        Returns:
            Tuple of (values, action_log_probs, entropy, logits):
                - values: Value predictions of shape (batch_size,)
                          (can be zeros if no critic network)
                - action_log_probs: Log probabilities of taken actions
                                    of shape (batch_size,)
                - entropy: Per-position entropy of shape (batch_size,)
                - logits: Action logits of shape (batch_size, action_dim)
        """
        pass


class GPT2Policy(BasePolicy):
    """GPT-2 based policy model for reinforcement learning.

    This policy uses a GPT-2 architecture (based on minGPT) to process
    continuous observations and output action probabilities. The observations
    are projected to the embedding dimension and processed through transformer
    blocks with causal self-attention.

    The architecture follows the minGPT implementation from
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py, with
    modifications to accept continuous observations instead of discrete tokens.

    Args:
        action_space: The action space (gymnasium.spaces.Discrete)
        observation_space: The observation space (gymnasium.spaces.Box)
        n_layer: Number of transformer layers. Default: 4
        n_head: Number of attention heads. Default: 4
        n_embd: Embedding dimension. Default: 128
        block_size: Maximum sequence length. Default: 64
        embd_pdrop: Embedding dropout rate. Default: 0.1
        resid_pdrop: Residual dropout rate. Default: 0.1
        attn_pdrop: Attention dropout rate. Default: 0.1

    Example:
        >>> from gymnasium import spaces
        >>> action_space = spaces.Discrete(10)
        >>> observation_space = spaces.Box(low=-1, high=1, shape=(768,))
        >>> policy = GPT2Policy(action_space, observation_space, n_layer=6, n_embd=256)
        >>> obs = torch.randn(8, 768)  # batch=8, dim=768
        >>> logits = policy(obs)  # (8, 10)
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 64,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ):
        super().__init__(action_space, observation_space)

        self.n_embd = n_embd
        self.block_size = block_size
        self.action_dim = int(action_space.n)

        # Project observations to embedding dimension
        obs_dim = observation_space.shape[0]
        self.obs_projection = nn.Linear(obs_dim, n_embd)

        # Positional embeddings (max block_size positions)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)

        # Dropout
        self.drop = nn.Dropout(embd_pdrop)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, embd_pdrop, resid_pdrop, attn_pdrop)
                for _ in range(n_layer)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Action head - maps embeddings to action logits
        self.action_head = nn.Linear(n_embd, self.action_dim)

        # Initialize weights following GPT-2 conventions
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 paper conventions.

        - Linear layers: normal initialization with std=0.02
        - Embedding layers: normal initialization with std=0.02
        - LayerNorm layers: bias=0, weight=1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT-2 policy.

        Args:
            obs: Observations of shape (batch_size, obs_dim)

        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        # Add sequence dimension (seq_len=1)
        obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        batch_size, seq_len, _ = obs.shape

        # Check sequence length
        assert seq_len <= self.block_size, (
            f"Cannot forward sequence of length {seq_len}, "
            f"block_size is only {self.block_size}"
        )

        # Project observations to embedding dimension
        x = self.obs_projection(obs)  # (batch, seq_len, n_embd)

        # Add positional embeddings
        device = obs.device
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )  # (1, seq_len)
        pos_emb = self.pos_embeddings(pos)  # (1, seq_len, n_embd)
        x = x + pos_emb

        # Apply dropout
        x = self.drop(x)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Get action logits
        logits = self.action_head(x)  # (batch, seq_len, action_dim)

        # Remove sequence dimension
        logits = logits.squeeze(1)  # (batch_size, action_dim)

        return logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions taken under the policy.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            actions: Actions taken of shape (batch_size,)
            masks: Masks of shape (batch_size,)

        Returns:
            Tuple of (values, action_log_probs, entropy, logits):
                - values: Zeros (no critic network in REINFORCE++) of shape (batch_size,)
                - action_log_probs: Log probs of taken actions of shape (batch_size,)
                - entropy: Per-position entropy of shape (batch_size,)
                - logits: Action logits of shape (batch_size, action_dim)
        """
        # Get action logits
        logits = self.forward(obs)  # (batch_size, action_dim)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, action_dim)

        # Get log probabilities of taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size,)

        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size,)

        # No value predictions in REINFORCE++ (no critic network)
        values = torch.zeros_like(actions, dtype=torch.float32, device=obs.device)

        return values, action_log_probs, entropy, logits


class Block(nn.Module):
    """A transformer block with causal self-attention and MLP.

    This is based on the Block class from minGPT. It consists of:
    1. LayerNorm + Causal Self-Attention + residual connection
    2. LayerNorm + MLP (GELU activation) + residual connection

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        embd_pdrop: Dropout for attention output
        resid_pdrop: Dropout for residual connections
        attn_pdrop: Dropout for attention weights
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        embd_pdrop: float,
        resid_pdrop: float,
        attn_pdrop: float,
    ):
        super().__init__()
        assert n_embd % n_head == 0

        # Causal self-attention
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop)

        # Layer norms
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        # MLP with GELU activation
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention layer.

    This implements multi-head self-attention with a causal mask to ensure
    that each position can only attend to previous positions (leftward).
    Based on the implementation from minGPT.

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        attn_pdrop: Dropout rate for attention weights
    """

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float):
        super().__init__()
        assert n_embd % n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(attn_pdrop)
        # Number of heads and embedding dim per head
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Attend to all preceding positions (causal mask)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        causal_mask = self._get_causal_mask(T, att.device, att.dtype)
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Aggregate values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _get_causal_mask(self, t: int, device: torch.device, dtype: torch.dtype):
        """Get causal mask for the current sequence length.

        Args:
            t: Current sequence length
            device: Device for the mask
            dtype: Data type for the mask

        Returns:
            Causal mask tensor of shape (1, 1, t, t)
        """
        # Create a triangular mask: 1 where i >= j (can attend), 0 otherwise
        mask = torch.tril(torch.ones(t, t, device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, t, t)


class NewGELU(nn.Module):
    """GELU activation function (identical to OpenAI GPT/BERT).

    Reference: Gaussian Error Linear Units (GELU) paper
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GELU activation.

        Args:
            x: Input tensor

        Returns:
            GELU-activated tensor
        """
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )
