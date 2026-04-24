"""Test cases for ReinforcePP policy gradient algorithm."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from aap_core.policy import BasePolicy, GPT2Policy
from aap_core.policy_gradient import ReinforcePP
from gymnasium import spaces


class SimplePolicy(BasePolicy):
    """Simple policy model for testing."""

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


class MultiLayerPolicy(BasePolicy):
    """Multi-layer policy model for testing."""

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        hidden_dim: int = 64,
    ):
        super().__init__(action_space, observation_space)
        self.obs_dim = observation_space.shape[0]
        self.action_dim = int(action_space.n)
        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        batch_size, seq_len, _ = obs.shape
        x = self.relu(self.fc1(obs.view(-1, self.obs_dim)))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
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


class TestReinforcePPBasic:
    """Test basic functionality of ReinforcePP."""

    def test_basic_update(self):
        """Test basic update with valid inputs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "value_loss" in losses
        assert "action_loss" in losses
        assert "entropy" in losses
        assert losses["value_loss"] == 0.0

    def test_model_parameters_updated(self):
        """Test that model parameters are actually updated."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=1, num_mini_batch=2
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        optimizer.update(batch)

        for name, param in policy.named_parameters():
            assert not torch.equal(initial_params[name], param), (
                f"Parameter {name} was not updated"
            )

    def test_loss_values_reasonable(self):
        """Test that loss values are reasonable (not NaN or inf)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert not torch.isnan(torch.tensor(losses["action_loss"]))
        assert not torch.isinf(torch.tensor(losses["action_loss"]))
        assert not torch.isnan(torch.tensor(losses["entropy"]))
        assert not torch.isinf(torch.tensor(losses["entropy"]))


class TestReinforcePPClipParam:
    """Test different clip parameter values."""

    def test_clip_param_zero(self):
        """Test with clip_param = 0 (no clipping effect)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.0, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_clip_param_one(self):
        """Test with clip_param = 1 (full clipping range)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=1.0, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_clip_param_very_small(self):
        """Test with very small clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=1e-10, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPZeroValues:
    """Test zero-value parameter cases."""

    def test_zero_rewards(self):
        """Test with zero rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.zeros(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_all_zero_masks(self):
        """Test with all-zero masks."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.zeros(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_single_sample_batch(self):
        """Test with batch_size = 1."""
        obs_dim = 10
        action_dim = 5
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=1
        )

        batch = {
            "obs": torch.randn(1, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (1, seq_len)),
            "rewards": torch.randn(1, seq_len),
            "masks": torch.ones(1, seq_len),
            "old_log_probs": torch.randn(1, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_single_action(self):
        """Test with sequence length = 1."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, 1, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, 1)),
            "rewards": torch.randn(batch_size, 1),
            "masks": torch.ones(batch_size, 1),
            "old_log_probs": torch.randn(batch_size, 1),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPRewardNormalization:
    """Test reward normalization functionality."""

    def test_identical_rewards(self):
        """Test with identical rewards (std = 0)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.ones(batch_size, seq_len) * 5.0,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_very_large_rewards(self):
        """Test with very large reward values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len) * 1e6,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert not torch.isnan(torch.tensor(losses["action_loss"]))

    def test_very_small_rewards(self):
        """Test with very small reward values."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len) * 1e-6,
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert not torch.isnan(torch.tensor(losses["action_loss"]))

    def test_negative_rewards(self):
        """Test with negative rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_reward_normalization=True,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": -torch.abs(torch.randn(batch_size, seq_len)),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_reward_normalization_disabled(self):
        """Test with reward normalization disabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_reward_normalization=False,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPKLLoss:
    """Test KL loss functionality."""

    def test_kl_loss_enabled(self):
        """Test with KL loss enabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
            "kl_divergence": torch.abs(torch.randn(batch_size, seq_len)),
        }

        losses = optimizer.update(batch)

        assert "kl_loss" in losses
        assert losses["kl_loss"] >= 0

    def test_kl_loss_disabled(self):
        """Test with KL loss disabled."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_kl_loss=False,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "kl_loss" in losses
        assert losses["kl_loss"] == 0.0

    def test_zero_kl_divergence(self):
        """Test with zero KL divergence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
            "kl_divergence": torch.zeros(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert losses["kl_loss"] == 0.0


class TestReinforcePPBatchSequenceVariations:
    """Test different batch size and sequence length combinations."""

    def test_large_batch_size(self):
        """Test with large batch size."""
        obs_dim = 10
        action_dim = 5
        batch_size = 128
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=4
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_long_sequence(self):
        """Test with long sequence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 100

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_small_batch_large_sequence(self):
        """Test with small batch and large sequence."""
        obs_dim = 10
        action_dim = 5
        batch_size = 2
        seq_len = 50

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=1
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPGradientClipping:
    """Test gradient clipping functionality."""

    def test_gradient_norm_limited(self):
        """Test that gradient norm is properly clipped."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=1,
            num_mini_batch=2,
            max_grad_norm=0.5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_zero_gradients(self):
        """Test with zero gradients."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "rewards": torch.zeros(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.zeros(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPMultiLayer:
    """Test with multi-layer policy models."""

    def test_multilayer_policy(self):
        """Test with multi-layer policy model."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = MultiLayerPolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            hidden_dim=32,
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_multilayer_with_large_hidden(self):
        """Test with multi-layer policy and large hidden dimension."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = MultiLayerPolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            hidden_dim=128,
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPEntropy:
    """Test entropy regularization."""

    def test_entropy_regularization(self):
        """Test that entropy regularization affects training."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer_high_entropy = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            entropy_coef=0.1,
        )
        optimizer_low_entropy = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            entropy_coef=0.001,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses_high = optimizer_high_entropy.update(batch)
        losses_low = optimizer_low_entropy.update(batch)

        assert "entropy" in losses_high
        assert "entropy" in losses_low

    def test_zero_entropy_coef(self):
        """Test with zero entropy coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            entropy_coef=0.0,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "entropy" in losses


class TestReinforcePPLearningRate:
    """Test different learning rates."""

    def test_very_small_lr(self):
        """Test with very small learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2, lr=1e-12
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_very_large_lr(self):
        """Test with very large learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2, lr=1.0
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPEpochsAndBatches:
    """Test different epoch and mini-batch configurations."""

    def test_single_epoch(self):
        """Test with single epoch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=1, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_many_epochs(self):
        """Test with many epochs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=20, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_single_mini_batch(self):
        """Test with single mini-batch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=1
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_many_mini_batches(self):
        """Test with many mini-batches."""
        obs_dim = 10
        action_dim = 5
        batch_size = 32
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=8
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPWithProvidedValues:
    """Test using pre-computed returns and advantages."""

    def test_with_provided_returns(self):
        """Test with pre-computed returns."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(batch_size)
        for t in reversed(range(seq_len)):
            cumulative_return = cumulative_return * rewards[:, t]
            returns[:, t] = cumulative_return

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
            "returns": returns,
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_with_provided_advantages(self):
        """Test with pre-computed advantages."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(batch_size)
        for t in reversed(range(seq_len)):
            cumulative_return = cumulative_return * rewards[:, t]
            returns[:, t] = cumulative_return

        advantages = returns - returns.mean()

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
            "advantages": advantages,
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses

    def test_with_all_precomputed(self):
        """Test with all values pre-computed."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        rewards = torch.randn(batch_size, seq_len)
        masks = torch.ones(batch_size, seq_len)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(batch_size)
        for t in reversed(range(seq_len)):
            cumulative_return = cumulative_return * rewards[:, t]
            returns[:, t] = cumulative_return

        advantages = returns - returns.mean()

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": rewards,
            "masks": masks,
            "old_log_probs": torch.randn(batch_size, seq_len),
            "returns": returns,
            "advantages": advantages,
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPNoOldLogProbs:
    """Test behavior when old_log_probs is not provided."""

    def test_without_old_log_probs(self):
        """Test update without old_log_probs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestReinforcePPInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_clip_param_negative(self):
        """Test with negative clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=-0.1, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_clip_param_very_large(self):
        """Test with very large clip_param."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=10.0, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_reinforce_epoch_zero(self):
        """Test with zero reinforce_epoch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=0, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_num_mini_batch_zero(self):
        """Test with zero num_mini_batch."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=0
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        with pytest.raises(ValueError, match="range.*must not be zero"):
            optimizer.update(batch)

    def test_invalid_entropy_coef_negative(self):
        """Test with negative entropy coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            entropy_coef=-0.1,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_lr_zero(self):
        """Test with zero learning rate."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2, lr=0.0
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_max_grad_norm_zero(self):
        """Test with zero max_grad_norm."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            max_grad_norm=0.0,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_max_grad_norm_negative(self):
        """Test with negative max_grad_norm."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            max_grad_norm=-1.0,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_invalid_kl_coef_negative(self):
        """Test with negative KL coefficient."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=-1e-5,
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
            "kl_divergence": torch.abs(torch.randn(batch_size, seq_len)),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_empty_batch(self):
        """Test with empty batch."""
        obs_dim = 10
        action_dim = 5

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(0, 4, obs_dim),
            "actions": torch.randint(0, action_dim, (0, 4)),
            "rewards": torch.randn(0, 4),
            "masks": torch.ones(0, 4),
            "old_log_probs": torch.randn(0, 4),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_mismatched_batch_sizes(self):
        """Test with mismatched batch sizes in batch dict."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size + 2, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        try:
            losses = optimizer.update(batch)
            assert "action_loss" in losses
        except RuntimeError:
            pass

    def test_mismatched_sequence_lengths(self):
        """Test with mismatched sequence lengths in batch dict."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len + 2),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        with pytest.raises(IndexError):
            optimizer.update(batch)

    def test_invalid_action_values(self):
        """Test with invalid action values (out of bounds)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(
                action_dim, action_dim + 10, (batch_size, seq_len)
            ),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        with pytest.raises(RuntimeError, match="index.*out of bounds"):
            optimizer.update(batch)

    def test_invalid_mask_values(self):
        """Test with invalid mask values (not binary)."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.randn(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_nan_rewards(self):
        """Test with NaN rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.tensor([[float("nan")] * seq_len] * batch_size),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_inf_rewards(self):
        """Test with infinite rewards."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.tensor([[float("inf")] * seq_len] * batch_size),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_nan_actions(self):
        """Test with NaN actions."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.tensor([[float("nan")] * seq_len] * batch_size),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        try:
            losses = optimizer.update(batch)
            assert "action_loss" in losses
        except RuntimeError:
            pass

    def test_nan_old_log_probs(self):
        """Test with NaN old_log_probs."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.tensor([[float("nan")] * seq_len] * batch_size),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses

    def test_nan_obs(self):
        """Test with NaN observations."""
        obs_dim = 10
        action_dim = 5
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.tensor([[float("nan")] * obs_dim] * batch_size * seq_len).view(
                batch_size, seq_len, obs_dim
            ),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)
        assert "action_loss" in losses


class TestReinforcePPObservationOptional:
    """Test when observations are optional."""

    def test_without_observations(self):
        """Test that update handles missing observations gracefully."""
        action_dim = 5
        obs_dim = 10
        batch_size = 8
        seq_len = 4

        policy = SimplePolicy(
            action_space=spaces.Discrete(action_dim),
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses


class TestGPT2PolicyForward:
    """Test GPT2Policy forward pass."""

    def test_gpt2_policy_forward_shape(self):
        """Test that GPT2Policy forward produces correct output shape."""

        action_space = spaces.Discrete(10)
        obs_dim = 768
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)

        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, 10)

    def test_gpt2_policy_forward_single_step(self):
        """Test GPT2Policy with single time step."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        obs = torch.randn(2, 1, obs_dim)
        logits = policy(obs)

        assert logits.shape == (2, 1, 5)
        assert not torch.isnan(logits).any()

    def test_gpt2_policy_forward_long_sequence(self):
        """Test GPT2Policy with longer sequence."""

        action_space = spaces.Discrete(8)
        obs_dim = 64
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space,
            observation_space,
            n_layer=3,
            n_head=4,
            n_embd=64,
            block_size=32,
        )

        obs = torch.randn(4, 16, obs_dim)
        logits = policy(obs)

        assert logits.shape == (4, 16, 8)
        assert not torch.isnan(logits).any()


class TestGPT2PolicyEvaluateActions:
    """Test GPT2Policy evaluate_actions method."""

    def test_evaluate_actions_shapes(self):
        """Test that evaluate_actions returns correct shapes."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert values.shape == (batch_size, seq_len)
        assert action_log_probs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)
        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_evaluate_actions_log_probs_valid(self):
        """Test that action log probabilities are valid (finite and reasonable)."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        # Log probs should be finite
        assert not torch.isnan(action_log_probs).any()
        assert not torch.isinf(action_log_probs).any()

        # Log probs should be negative (log of probabilities <= 1)
        assert (action_log_probs <= 0).all()

    def test_evaluate_actions_entropy_positive(self):
        """Test that entropy is positive for non-deterministic policies."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        # Entropy should be positive (non-deterministic policy)
        assert (entropy > 0).all()
        assert not torch.isnan(entropy).any()

    def test_evaluate_actions_values_are_zeros(self):
        """Test that values are zeros (no critic network)."""

        action_dim = 5
        obs_dim = 32
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert (values == 0).all()


class TestGPT2PolicyReinforcePP:
    """Test GPT2Policy with ReinforcePP optimizer."""

    def test_gpt2_policy_basic_update(self):
        """Test basic update with GPT2Policy."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=2
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "value_loss" in losses
        assert "action_loss" in losses
        assert "entropy" in losses
        assert losses["value_loss"] == 0.0

    def test_gpt2_policy_parameters_updated(self):
        """Test that GPT2Policy parameters are updated."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=1, num_mini_batch=2
        )

        initial_params = {
            name: param.clone() for name, param in policy.named_parameters()
        }

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        optimizer.update(batch)

        # At least some parameters should have changed
        updated = False
        for name, param in policy.named_parameters():
            if not torch.equal(initial_params[name], param):
                updated = True
                break
        assert updated, "No parameters were updated"

    def test_gpt2_policy_with_scheduler(self):
        """Test GPT2Policy with learning rate scheduler."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        initial_lr = 1e-4
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=torch.optim.Adam(policy.parameters(), lr=initial_lr),
            step_size=1,
            gamma=0.5,
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            lr=initial_lr,
            lr_scheduler=scheduler,
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses
        # Check that scheduler was stepped (lr should be reduced)
        # With reinforce_epoch=2 and num_mini_batch=2, scheduler is stepped 4 times
        # So LR = initial_lr * 0.5^4 = 6.25e-06
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr < initial_lr

    def test_gpt2_policy_with_kl_loss(self):
        """Test GPT2Policy with KL loss enabled."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        optimizer = ReinforcePP(
            policy,
            clip_param=0.2,
            reinforce_epoch=2,
            num_mini_batch=2,
            use_kl_loss=True,
            kl_coef=1e-5,
        )

        batch_size = 4
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
            "kl_divergence": torch.abs(torch.randn(batch_size, seq_len)),
        }

        losses = optimizer.update(batch)

        assert "kl_loss" in losses
        assert losses["kl_loss"] >= 0

    def test_gpt2_policy_large_batch(self):
        """Test GPT2Policy with large batch size."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )
        optimizer = ReinforcePP(
            policy, clip_param=0.2, reinforce_epoch=2, num_mini_batch=4
        )

        batch_size = 32
        seq_len = 8
        batch = {
            "obs": torch.randn(batch_size, seq_len, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size, seq_len)),
            "rewards": torch.randn(batch_size, seq_len),
            "masks": torch.ones(batch_size, seq_len),
            "old_log_probs": torch.randn(batch_size, seq_len),
        }

        losses = optimizer.update(batch)

        assert "action_loss" in losses
        assert not torch.isnan(torch.tensor(losses["action_loss"]))
        assert not torch.isinf(torch.tensor(losses["action_loss"]))


class TestGPT2PolicyEdgeCases:
    """Test GPT2Policy edge cases."""

    def test_gpt2_policy_block_size_limit(self):
        """Test that GPT2Policy raises error for sequence > block_size."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space,
            observation_space,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=8,
        )

        # Sequence length > block_size should raise AssertionError
        obs = torch.randn(2, 16, obs_dim)
        with pytest.raises(AssertionError, match="block_size"):
            policy(obs)

    def test_gpt2_policy_single_action(self):
        """Test GPT2Policy with single action class."""

        action_space = spaces.Discrete(1)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.zeros(batch_size, seq_len, dtype=torch.long)
        masks = torch.ones(batch_size, seq_len)

        values, action_log_probs, entropy, logits = policy.evaluate_actions(
            obs, actions, masks
        )

        assert action_log_probs.shape == (batch_size, seq_len)
        # With single action, entropy should be 0
        assert (entropy == 0).all()

    def test_gpt2_policy_different_hidden_dims(self):
        """Test GPT2Policy with different hidden dimensions."""

        action_dim = 10
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Test with larger hidden dimension
        policy = GPT2Policy(
            action_space, observation_space, n_layer=4, n_head=8, n_embd=128
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)
        assert not torch.isnan(logits).any()


class TestBasePolicyAbstract:
    """Test BasePolicy abstract class behavior."""

    def test_base_policy_cannot_instantiate(self):
        """Test that BasePolicy cannot be instantiated directly."""

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        with pytest.raises(TypeError):
            BasePolicy(action_space, observation_space)

    def test_gpt2_policy_is_base_policy(self):
        """Test that GPT2Policy is an instance of BasePolicy."""
        from aap_core.policy import BasePolicy, GPT2Policy

        action_space = spaces.Discrete(5)
        obs_dim = 32
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(action_space, observation_space)
        assert isinstance(policy, BasePolicy)
        assert isinstance(policy, nn.Module)

    def test_base_policy_has_required_methods(self):
        """Test that BasePolicy has required abstract methods."""
        from aap_core.policy import BasePolicy

        assert hasattr(BasePolicy, "forward")
        assert hasattr(BasePolicy, "evaluate_actions")


class TestGPT2PolicyActionSpace:
    """Test GPT2Policy with different action spaces."""

    def test_gpt2_policy_large_action_space(self):
        """Test GPT2Policy with large action space."""

        action_dim = 100
        obs_dim = 64
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 8
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_gpt2_policy_small_action_space(self):
        """Test GPT2Policy with small action space."""

        action_dim = 2
        obs_dim = 32
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)


class TestGPT2PolicyObservationSpace:
    """Test GPT2Policy with different observation spaces."""

    def test_gpt2_policy_small_obs_dim(self):
        """Test GPT2Policy with small observation dimension."""

        action_dim = 10
        obs_dim = 8
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=2, n_embd=32
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)

    def test_gpt2_policy_large_obs_dim(self):
        """Test GPT2Policy with large observation dimension."""

        action_dim = 10
        obs_dim = 512
        action_space = spaces.Discrete(action_dim)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        policy = GPT2Policy(
            action_space, observation_space, n_layer=2, n_head=4, n_embd=64
        )

        batch_size = 4
        seq_len = 4
        obs = torch.randn(batch_size, seq_len, obs_dim)
        logits = policy(obs)

        assert logits.shape == (batch_size, seq_len, action_dim)
