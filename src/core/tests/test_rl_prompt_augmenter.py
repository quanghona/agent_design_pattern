#!/usr/bin/env python3
"""Comprehensive tests for RLPromptAugmenter class.

Tests cover __init__, train, and augment methods with normal inputs,
edge cases, extreme values, and exception scenarios.
"""

import os
import shutil
import sys
import tempfile
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

# Ensure the core package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aap_core.policy import BasePolicy, GPT2Policy
from aap_core.prompt_augmenter import (
    IdentityPromptAugmenter,
    PromptOptimizationEnv,
    RLPromptAugmenter,
    SimplePromptAugmenter,
)
from aap_core.types import AgentMessage


class SimplePolicy(BasePolicy):
    """Simple single-layer policy model for testing."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
    ):
        super().__init__(action_space, observation_space)
        self.action_dim = int(action_space.n)
        self.obs_dim = observation_space.shape[0]
        self.fc = nn.Linear(self.obs_dim, self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        batch_size, seq_len, _ = obs.shape
        logits = self.fc(obs.view(-1, self.obs_dim))
        return logits.view(batch_size, seq_len, self.action_dim)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Evaluate actions and return values."""
        logits = self.forward(obs)
        batch_size, seq_len, _ = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        values = torch.zeros(batch_size, seq_len, device=obs.device)
        return values, action_log_probs, entropy, logits


def _create_dummy_state_dict(n_layer=4, n_embd=64, n_head=4):
    """Create a complete dummy state dict for GPT2Policy."""
    state_dict = {
        "obs_projection.weight": torch.randn(n_embd, n_embd),
        "obs_projection.bias": torch.randn(n_embd),
        "pos_embeddings.weight": torch.randn(64, n_embd),
    }

    # Add blocks
    for i in range(n_layer):
        state_dict[f"blocks.{i}.attn.c_attn.weight"] = torch.randn(3 * n_embd, n_embd)
        state_dict[f"blocks.{i}.attn.c_attn.bias"] = torch.randn(3 * n_embd)
        state_dict[f"blocks.{i}.attn.c_proj.weight"] = torch.randn(n_embd, n_embd)
        state_dict[f"blocks.{i}.attn.c_proj.bias"] = torch.randn(n_embd)
        state_dict[f"blocks.{i}.ln_1.weight"] = torch.randn(n_embd)
        state_dict[f"blocks.{i}.ln_1.bias"] = torch.randn(n_embd)
        state_dict[f"blocks.{i}.ln_2.weight"] = torch.randn(n_embd)
        state_dict[f"blocks.{i}.ln_2.bias"] = torch.randn(n_embd)
        state_dict[f"blocks.{i}.mlp.c_fc.weight"] = torch.randn(4 * n_embd, n_embd)
        state_dict[f"blocks.{i}.mlp.c_fc.bias"] = torch.randn(4 * n_embd)
        state_dict[f"blocks.{i}.mlp.c_proj.weight"] = torch.randn(n_embd, 4 * n_embd)
        state_dict[f"blocks.{i}.mlp.c_proj.bias"] = torch.randn(n_embd)

    state_dict["ln_f.weight"] = torch.randn(n_embd)
    state_dict["ln_f.bias"] = torch.randn(n_embd)
    state_dict["action_head.weight"] = torch.randn(2, n_embd)
    state_dict["action_head.bias"] = torch.randn(2)

    return state_dict


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def embedding_model():
    """Return a simple embedding model that produces fixed-dimension vectors."""

    def model(prompt: str) -> np.ndarray:
        return np.random.randn(64).astype(np.float32)

    return model


@pytest.fixture
def reward_model():
    """Return a reward model that gives higher scores for longer prompts."""

    def model(prompt: str) -> float:
        return float(len(prompt)) / 100.0

    return model


@pytest.fixture
def augmenters():
    """Return a list of simple augmenters for the environment."""
    return [
        IdentityPromptAugmenter(),
        SimplePromptAugmenter(format="{query} {data}"),
    ]


@pytest.fixture
def env(embedding_model, reward_model, augmenters):
    """Return a PromptOptimizationEnv with minimal configuration."""
    return PromptOptimizationEnv(
        initial_prompt="test prompt",
        augmenters=augmenters,
        embedding_model=embedding_model,
        reward_model=reward_model,
        max_steps=3,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for model checkpoints."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def policy_model(env):
    """Return a simple policy model for inference tests."""
    return SimplePolicy(env.action_space, env.observation_space)


@pytest.fixture
def rl_augmenter(env, policy_model):
    """Return an RLPromptAugmenter configured for inference."""
    return RLPromptAugmenter(env=env, policy_model=policy_model)


# ──────────────────────────────────────────────
# Tests for __init__
# ──────────────────────────────────────────────


class TestRLPromptAugmenterInit:
    """Tests for RLPromptAugmenter.__init__."""

    def test_init_with_policy_model(self, env, policy_model):
        """Normal initialization with an existing policy model."""
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        assert aug.env is env
        assert aug.policy_model is policy_model

    def test_init_requires_policy_model(self, env):
        """RLPromptAugmenter requires a policy model at initialization."""
        with pytest.raises(ValueError):
            RLPromptAugmenter(env=env)


# ──────────────────────────────────────────────
# Tests for train()
# ──────────────────────────────────────────────


class TestRLPromptAugmenterTrain:
    """Tests to confirm RLPromptAugmenter is inference-only."""

    def test_train_method_does_not_exist(self, rl_augmenter):
        """RLPromptAugmenter should not expose a training API."""
        with pytest.raises(AttributeError):
            rl_augmenter.train()


# ──────────────────────────────────────────────
# Tests for augment()
# ──────────────────────────────────────────────


class TestRLPromptAugmenterAugment:
    """Tests for RLPromptAugmenter.augment()."""

    def test_augment_normal_inference(self, rl_augmenter):
        """Normal inference: augment returns modified message."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)
        assert hasattr(result, "query")

    def test_augment_infer_mode(self, rl_augmenter):
        """Inference path returns a valid AgentMessage."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)
        assert result.query == msg.query or isinstance(result.query, str)

    def test_augment_terminated_early(
        self, embedding_model, reward_model, augmenters
    ):
        """Augment handles episodes that terminate early."""

        def high_reward(prompt: str) -> float:
            return 1000.0

        env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=high_reward,
            max_steps=10,
            reward_threshold=100.0,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test query for augmentation")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_different_prompts(self, rl_augmenter):
        """Augment works with different initial prompts."""
        prompts = ["short", "a much longer prompt with more words to test", ""]
        for prompt in prompts:
            msg = AgentMessage(query=prompt)
            result = rl_augmenter.augment(msg)
            assert isinstance(result, AgentMessage)

    def test_augment_returns_same_message_object(self, rl_augmenter):
        """Augment modifies and returns the same message object."""
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter.augment(msg)
        assert result is msg

    def test_augment_with_kwargs_passed_through(self, rl_augmenter):
        """Augment accepts kwargs without failure."""
        msg = AgentMessage(query="test prompt")
        result = rl_augmenter.augment(msg, extra_arg="ignored")
        assert isinstance(result, AgentMessage)

    def test_augment_with_single_step_env(
        self, embedding_model, reward_model, augmenters
    ):
        """Augment with environment that has max_steps=1."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=1,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_preserves_message_attributes(self, rl_augmenter):
        """Augment preserves other message attributes beyond query."""
        msg = AgentMessage(
            query="original",
            origin="test_agent",
            context={"key": "value"},
        )
        result = rl_augmenter.augment(msg)
        assert result.origin == "test_agent"
        assert result.context == {"key": "value"}


# ──────────────────────────────────────────────
# Integration / Edge Case Tests
# ──────────────────────────────────────────────


class TestRLPromptAugmenterIntegration:
    """Integration and edge case tests for RLPromptAugmenter."""

    def test_infer_with_single_augmenter(self, embedding_model, reward_model):
        """Environment with only one augmenter works with inference."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_infer_with_many_augmenters(self, embedding_model, reward_model):
        """Environment with many augmenters works with inference."""
        augmenters = [IdentityPromptAugmenter() for _ in range(10)]
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        policy_model = SimplePolicy(env.action_space, env.observation_space)
        aug = RLPromptAugmenter(env=env, policy_model=policy_model)
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_empty_query(self, rl_augmenter):
        """Augment with an empty query string."""
        msg = AgentMessage(query="")
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_unicode_query(self, rl_augmenter):
        """Augment with unicode characters in query."""
        msg = AgentMessage(
            query="Hello 世界 \U0001f30d \u041f\u0440\u0438\u0432\u0435\u0442"
        )
        result = rl_augmenter.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_env_with_zero_augmenters_raises(self, embedding_model, reward_model):
        """Environment with 0 augmenters should raise AssertionError."""
        with pytest.raises(AssertionError, match="At least one augmenter must be provided"):
            PromptOptimizationEnv(
                initial_prompt="test",
                augmenters=[],
                embedding_model=embedding_model,
                reward_model=reward_model,
                max_steps=2,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
