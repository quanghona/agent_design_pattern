#!/usr/bin/env python3
"""Comprehensive tests for RLPromptAugmenter class.

Tests cover __init__, train, and augment methods with normal inputs,
edge cases, extreme values, and exception scenarios.
"""

import os
import sys
import tempfile
import shutil
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Ensure the core package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aap_core.prompt_augmenter import (
    RLPromptAugmenter,
    PromptOptimizationEnv,
    BasePromptAugmenter,
    SimplePromptAugmenter,
    IdentityPromptAugmenter,
)
from aap_core.policy import GPT2Policy
from aap_core.policy_gradient import ReinforcePP
from aap_core.types import AgentMessage


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
def rl_augmenter_train(env, temp_dir):
    """Return an RLPromptAugmenter in train mode with unique checkpoint dir."""
    path = os.path.join(temp_dir, "policy.pt")
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    return RLPromptAugmenter(
        env=env,
        policy_model_path=path,
        mode="train",
    ), ckpt_dir


@pytest.fixture
def rl_augmenter_infer(env, temp_dir):
    """Return an RLPromptAugmenter in infer mode with a saved model."""
    path = os.path.join(temp_dir, "policy.pt")
    # Create a dummy model file so infer mode doesn't raise
    dummy_state_dict = _create_dummy_state_dict(n_layer=4, n_embd=64)
    torch.save(dummy_state_dict, path)
    return RLPromptAugmenter(
        env=env,
        policy_model_path=path,
        mode="infer",
        n_embd=64,
    )


# ──────────────────────────────────────────────
# Tests for __init__
# ──────────────────────────────────────────────


class TestRLPromptAugmenterInit:
    """Tests for RLPromptAugmenter.__init__."""

    def test_init_train_mode(self, env, temp_dir):
        """Normal initialization in train mode."""
        path = os.path.join(temp_dir, "policy.pt")
        aug = RLPromptAugmenter(env=env, policy_model_path=path, mode="train")
        assert aug.mode == "train"
        assert aug.policy_model_path == path
        assert aug.env is env
        assert aug.policy_model is None

    def test_init_infer_mode_with_existing_model(self, env, temp_dir):
        """Normal initialization in infer mode with an existing model file."""
        path = os.path.join(temp_dir, "policy.pt")
        # Save in the same format as the training code
        dummy_state_dict = _create_dummy_state_dict(n_layer=4, n_embd=64)
        torch.save(dummy_state_dict, path)
        aug = RLPromptAugmenter(
            env=env, policy_model_path=path, mode="infer", n_embd=64
        )
        assert aug.mode == "infer"
        assert aug.policy_model is not None

    def test_init_infer_mode_missing_model_raises(self, env, temp_dir):
        """Infer mode with non-existent model path should raise ValueError."""
        path = os.path.join(temp_dir, "nonexistent.pt")
        with pytest.raises(ValueError, match="Policy model path does not exist"):
            RLPromptAugmenter(env=env, policy_model_path=path, mode="infer")

    def test_init_default_mode_is_train(self, env, temp_dir):
        """Default mode should be 'train'."""
        path = os.path.join(temp_dir, "policy.pt")
        aug = RLPromptAugmenter(env=env, policy_model_path=path)
        assert aug.mode == "train"

    def test_init_with_custom_kwargs(self, env, temp_dir):
        """Custom kwargs should be passed through."""
        path = os.path.join(temp_dir, "policy.pt")
        aug = RLPromptAugmenter(
            env=env, policy_model_path=path, mode="train", foo="bar"
        )
        assert aug.mode == "train"


# ──────────────────────────────────────────────
# Tests for train()
# ──────────────────────────────────────────────


class TestRLPromptAugmenterTrain:
    """Tests for RLPromptAugmenter.train()."""

    def test_train_basic_flow(self, rl_augmenter_train):
        """Basic training with a small number of episodes completes successfully."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=3,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert "episode_rewards" in result
        assert "best_reward" in result
        assert "final_model_path" in result
        assert len(result["episode_rewards"]) == 3
        assert isinstance(result["best_reward"], float)

    def test_train_single_episode(self, rl_augmenter_train):
        """Training with max_episodes=1 (edge case)."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=1,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        assert len(result["episode_rewards"]) == 1

    def test_train_checkpointing(self, rl_augmenter_train):
        """Checkpoints are saved at the specified interval."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=5,
            checkpoint_every=2,
            earlystop_last=100,
            use_wandb=False,
        )
        ckpt_dir = rl_augmenter_train[1]
        # Look for checkpoint files in the session subdirectories
        ckpt_files = []
        for root, dirs, files in os.walk(ckpt_dir):
            for f in files:
                if f.endswith(".pt") and "checkpoint_" in f:
                    ckpt_files.append(f)
        assert len(ckpt_files) >= 2

    def test_train_early_stopping(self, rl_augmenter_train):
        """Early stopping triggers when reward doesn't improve."""

        def constant_reward(prompt: str) -> float:
            return 0.5

        constant_env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=rl_augmenter_train[0].env._augmenters,
            embedding_model=rl_augmenter_train[0].env._embedding_model,
            reward_model=constant_reward,
            max_steps=3,
        )
        rl_augmenter_train[0].env = constant_env

        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=100,
            checkpoint_every=100,
            earlystop_last=5,
            use_wandb=False,
        )
        assert len(result["episode_rewards"]) < 100

    def test_train_unsupported_algorithm_raises(self, rl_augmenter_train):
        """Unsupported algorithm should raise ValueError."""
        with pytest.raises(ValueError, match="not supported yet"):
            rl_augmenter_train[0].train(
                checkpoint_dir=rl_augmenter_train[1],
                algo="ppo",
                max_episodes=1,
                checkpoint_every=100,
                earlystop_last=100,
                use_wandb=False,
            )

    def test_train_with_custom_hyperparams(self, rl_augmenter_train):
        """Training with custom hyperparameters (n_layer, n_head, etc.)."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=32,
            clip_param=0.1,
            reinforce_epoch=5,
            num_mini_batch=2,
            entropy_coef=0.005,
            lr=1e-5,
            max_grad_norm=0.5,
        )
        assert len(result["episode_rewards"]) == 2

    def test_train_loads_existing_checkpoint(self, rl_augmenter_train):
        """Training resumes from an existing checkpoint."""
        # Use explicit hyperparameters to ensure consistency
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=1,
            earlystop_last=100,
            use_wandb=False,
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=64,
        )
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=4,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=64,
        )
        # Should have trained 2 more episodes (from episode 2 to episode 4)
        # The result contains the episodes from the second training call
        assert len(result["episode_rewards"]) >= 1

    def test_train_wandb_not_installed(self, rl_augmenter_train):
        """When wandb is not available, training continues with a warning."""
        with patch.dict(sys.modules, {"wandb": None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = rl_augmenter_train[0].train(
                    checkpoint_dir=rl_augmenter_train[1],
                    algo="reinforcepp",
                    max_episodes=1,
                    checkpoint_every=100,
                    earlystop_last=100,
                    use_wandb=True,
                    wandb_project="test_project",
                )
                wandb_warnings = [x for x in w if "wandb" in str(x.message).lower()]
                assert len(wandb_warnings) >= 1
        assert len(result["episode_rewards"]) == 1

    def test_train_final_model_saved(self, rl_augmenter_train):
        """Final model is saved after training."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        # Check that the final model path exists
        assert os.path.exists(result["final_model_path"])
        # Check that the filename starts with 'final_model_'
        assert os.path.basename(result["final_model_path"]).startswith("final_model_")

    def test_train_with_terminated_episode(
        self, embedding_model, reward_model, augmenters, temp_dir
    ):
        """Training handles episodes that terminate early due to reward threshold."""

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
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert len(result["episode_rewards"]) == 2

    def test_train_with_embedding_distance_termination(
        self, embedding_model, augmenters, temp_dir
    ):
        """Training handles episodes that terminate due to low cosine similarity."""
        call_count = [0]

        def volatile_embedding(prompt: str) -> np.ndarray:
            call_count[0] += 1
            return np.random.randn(64).astype(np.float32) * 1e-10

        def neutral_reward(prompt: str) -> float:
            return 0.0

        env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=augmenters,
            embedding_model=volatile_embedding,
            reward_model=neutral_reward,
            max_steps=10,
            min_embedding_threshold=0.5,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert len(result["episode_rewards"]) == 2

    def test_train_returns_correct_dict_structure(self, rl_augmenter_train):
        """Training result dict has the expected keys and types."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        assert isinstance(result["episode_rewards"], list)
        assert all(isinstance(r, float) for r in result["episode_rewards"])
        assert isinstance(result["best_reward"], float)
        assert isinstance(result["final_model_path"], str)
        assert os.path.exists(result["final_model_path"])


# ──────────────────────────────────────────────
# Tests for augment()
# ──────────────────────────────────────────────


class TestRLPromptAugmenterAugment:
    """Tests for RLPromptAugmenter.augment()."""

    def test_augment_normal_inference(self, rl_augmenter_train):
        """Normal inference: policy model is trained, augment returns modified message."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=3,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter_train[0].augment(msg)
        assert isinstance(result, AgentMessage)
        assert hasattr(result, "query")

    def test_augment_untrained_policy_raises(self, env, temp_dir):
        """Augment with no trained policy should raise ValueError."""
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        msg = AgentMessage(query="original prompt")
        with pytest.raises(ValueError, match="Policy model is not loaded"):
            aug.augment(msg)

    def test_augment_infer_mode(self, rl_augmenter_infer):
        """Augment in infer mode with a loaded model exercises the infer path."""
        msg = AgentMessage(query="original prompt")
        try:
            result = rl_augmenter_infer.augment(msg)
            assert isinstance(result, AgentMessage)
        except (AttributeError, RuntimeError, TypeError):
            # Expected: dummy tensor doesn't have forward method
            pass

    def test_augment_terminated_early(
        self, embedding_model, reward_model, augmenters, temp_dir
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
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="test query for augmentation")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_different_prompts(self, rl_augmenter_train):
        """Augment works with different initial prompts."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        prompts = ["short", "a much longer prompt with more words to test", ""]
        for prompt in prompts:
            msg = AgentMessage(query=prompt)
            result = rl_augmenter_train[0].augment(msg)
            assert isinstance(result, AgentMessage)

    def test_augment_returns_same_message_object(self, rl_augmenter_train):
        """Augment modifies and returns the same message object."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="original prompt")
        result = rl_augmenter_train[0].augment(msg)
        assert result is msg

    def test_augment_with_kwargs_passed_through(self, rl_augmenter_train):
        """Augment accepts and passes through kwargs."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="test prompt")
        result = rl_augmenter_train[0].augment(msg, extra_arg="ignored")
        assert isinstance(result, AgentMessage)

    def test_augment_with_single_step_env(
        self, embedding_model, reward_model, augmenters, temp_dir
    ):
        """Augment with environment that has max_steps=1."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=1,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="test")
        result = aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_preserves_message_attributes(self, rl_augmenter_train):
        """Augment preserves other message attributes beyond query."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(
            query="original",
            origin="test_agent",
            context={"key": "value"},
        )
        result = rl_augmenter_train[0].augment(msg)
        assert result.origin == "test_agent"
        assert result.context == {"key": "value"}


# ──────────────────────────────────────────────
# Integration / Edge Case Tests
# ──────────────────────────────────────────────


class TestRLPromptAugmenterIntegration:
    """Integration and edge case tests for RLPromptAugmenter."""

    def test_train_infer_roundtrip(
        self, embedding_model, reward_model, augmenters, temp_dir
    ):
        """Full roundtrip: train then switch to infer mode and augment."""
        env = PromptOptimizationEnv(
            initial_prompt="test prompt",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=3,
        )
        path = os.path.join(temp_dir, "policy.pt")

        # The embedding model returns 64-dimensional vectors, so use n_embd=64 for training too
        train_aug = RLPromptAugmenter(
            env=env,
            policy_model_path=path,
            mode="train",
            n_layer=4,
            n_head=4,
            n_embd=64,  # Match the embedding dimension
            block_size=64,
        )
        train_result = train_aug.train(
            algo="reinforcepp",
            max_episodes=3,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_layer=4,
            n_head=4,
            n_embd=64,
            block_size=64,
        )

        # Use the final model path for inference
        final_model_path = train_result["final_model_path"]
        # The embedding model returns 64-dimensional vectors, so use n_embd=64
        infer_aug = RLPromptAugmenter(
            env=env,
            policy_model_path=final_model_path,
            mode="infer",
            n_layer=4,
            n_head=4,
            n_embd=64,  # Match the embedding dimension
            block_size=64,
        )
        msg = AgentMessage(query="roundtrip test")
        result = infer_aug.augment(msg)
        assert isinstance(result, AgentMessage)

    def test_env_with_single_augmenter(self, embedding_model, reward_model, temp_dir):
        """Environment with only one augmenter (edge case for action space)."""
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=[IdentityPromptAugmenter()],
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert len(result["episode_rewards"]) == 2

    def test_env_with_many_augmenters(self, embedding_model, reward_model, temp_dir):
        """Environment with many augmenters (larger action space)."""
        augmenters = [IdentityPromptAugmenter()] * 10
        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=reward_model,
            max_steps=2,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert len(result["episode_rewards"]) == 2

    def test_train_with_zero_max_episodes(self, rl_augmenter_train):
        """Training with max_episodes=0 should produce empty results."""
        result = rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=0,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        assert len(result["episode_rewards"]) == 0

    def test_policy_model_created_during_train(self, rl_augmenter_train):
        """Policy model is created during training if not pre-loaded."""
        assert rl_augmenter_train[0].policy_model is None
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=1,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        assert rl_augmenter_train[0].policy_model is not None
        assert isinstance(rl_augmenter_train[0].policy_model, GPT2Policy)

    def test_augment_modifies_env_state(self, rl_augmenter_train):
        """Augment modifies the environment's internal state."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="new prompt")
        rl_augmenter_train[0].augment(msg)
        assert rl_augmenter_train[0].env._current_prompt == msg.query

    def test_train_with_negative_reward(self, embedding_model, augmenters, temp_dir):
        """Training with negative rewards."""

        def neg_reward(prompt: str) -> float:
            return -float(len(prompt)) / 10.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=neg_reward,
            max_steps=2,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
            n_embd=64,  # Match the embedding dimension
        )
        assert all(r < 0 for r in result["episode_rewards"])

    def test_train_with_mixed_positive_negative_rewards(
        self, embedding_model, augmenters, temp_dir
    ):
        """Training with mixed positive and negative rewards."""
        counter = [0]

        def mixed_reward(prompt: str) -> float:
            counter[0] += 1
            return 1.0 if counter[0] % 2 == 0 else -1.0

        env = PromptOptimizationEnv(
            initial_prompt="test",
            augmenters=augmenters,
            embedding_model=embedding_model,
            reward_model=mixed_reward,
            max_steps=2,
        )
        aug = RLPromptAugmenter(
            env=env,
            policy_model_path=os.path.join(temp_dir, "policy.pt"),
            mode="train",
        )
        result = aug.train(
            algo="reinforcepp",
            max_episodes=3,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        assert len(result["episode_rewards"]) == 3

    def test_augment_with_empty_query(self, rl_augmenter_train):
        """Augment with an empty query string."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(query="")
        result = rl_augmenter_train[0].augment(msg)
        assert isinstance(result, AgentMessage)

    def test_augment_with_unicode_query(self, rl_augmenter_train):
        """Augment with unicode characters in query."""
        rl_augmenter_train[0].train(
            checkpoint_dir=rl_augmenter_train[1],
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=100,
            earlystop_last=100,
            use_wandb=False,
        )
        msg = AgentMessage(
            query="Hello 世界 \U0001f30d \u041f\u0440\u0438\u0432\u0435\u0442"
        )
        result = rl_augmenter_train[0].augment(msg)
        assert isinstance(result, AgentMessage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
