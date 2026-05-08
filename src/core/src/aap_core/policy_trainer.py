import glob
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from aap_core.policy import BasePolicy
from aap_core.prompt_augmenter import PromptOptimizationEnv
from aap_core.types import AgentMessage


class BaseExplorationModule(ABC):
    """Abstract base class for exploration strategies in policy training.

    This class defines the interface for exploration modules that determine
    whether the policy should explore (sample stochastically) or exploit
    (select greedy actions) at each step during episode rollout.

    The exploration module is used during the ``fit()`` method's environment
    interaction loop, NOT during the gradient update step.

    Example usage with a custom module::

        class MyExploration(BaseExplorationModule):
            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return torch.rand(1).item() > 0.5

            def get_exploration_config(self) -> dict:
                return {"strategy": "my_custom"}

    Example usage with torchRL::

        from torchrl.envs import EpsilonGreedyActionSelection

        class TorchRLWrapper(BaseExplorationModule):
            def __init__(self, torchrl_exploration):
                self._exploration = torchrl_exploration

            def should_explore(self, step: int, episode: int, **kwargs) -> bool:
                return self._exploration.step()

            def get_exploration_config(self) -> dict:
                return {"source": "torchrl"}

    """

    @abstractmethod
    def should_explore(self, step: int, episode: int, **kwargs) -> bool:
        """Determine whether to explore at the current step.

        Args:
            step: Current step index within the episode (0-based).
            episode: Current episode index (0-based).
            **kwargs: Additional context (e.g., ``'obs'``, ``'logits'``, ``'info'``).

        Returns:
            ``True`` if the policy should explore (sample stochastically),
            ``False`` if it should exploit (select greedy action).
        """
        pass

    @abstractmethod
    def get_exploration_config(self) -> Dict[str, Any]:
        """Return a dictionary of exploration hyperparameters for logging.

        Returns:
            Dictionary of configuration parameters (e.g., epsilon, temperature).
        """
        pass

    def reset(self, episode: int) -> None:
        """Reset exploration state at the beginning of an episode.

        Override this method if the exploration strategy needs per-episode
        state (e.g., resetting an epsilon schedule).

        Args:
            episode: Current episode index (0-based).
        """
        pass


class EpsilonGreedyExploration(BaseExplorationModule):
    """Epsilon-greedy exploration with linear decay.

    Explores with probability epsilon, exploits with probability 1-epsilon.
    Epsilon decays linearly from ``eps_init`` to ``eps_final`` over
    ``decay_episodes``.

    Args:
        eps_init: Initial exploration probability (default ``1.0``).
        eps_final: Final exploration probability after decay (default ``0.05``).
        decay_episodes: Number of episodes over which to decay epsilon
            (default ``1000``).
        min_eps: Minimum epsilon floor (default ``0.01``).
    """

    def __init__(
        self,
        eps_init: float = 1.0,
        eps_final: float = 0.05,
        decay_episodes: int = 1000,
        min_eps: float = 0.01,
    ) -> None:
        self.eps_init = eps_init
        self.eps_final = max(eps_final, min_eps)
        self.decay_episodes = max(1, decay_episodes)
        self.current_eps: float = eps_init

    def should_explore(self, step: int, episode: int, **kwargs) -> bool:
        # Linear decay
        progress = min(episode / self.decay_episodes, 1.0)
        self.current_eps = self.eps_final + (self.eps_init - self.eps_final) * (
            1.0 - progress
        )
        self.current_eps = max(self.current_eps, self.eps_final)
        return torch.rand(1).item() < self.current_eps

    def get_exploration_config(self) -> Dict[str, Any]:
        return {
            "eps_init": self.eps_init,
            "eps_final": self.eps_final,
            "decay_episodes": self.decay_episodes,
            "current_eps": self.current_eps,
        }


class RandomExploration(EpsilonGreedyExploration):
    """Convenience class for constant explore/exploit ratio (no decay).

    This is a special case of EpsilonGreedyExploration where ``eps_init``
    equals ``eps_final``, resulting in a constant exploration probability
    regardless of the episode number. Useful for fixed exploration ratios.

    Args:
        explore_ratio: Probability of exploring (default ``0.0``).
    """

    def __init__(self, explore_ratio: float = 0.0) -> None:
        # Initialize with no decay: eps_init = eps_final = explore_ratio
        # Pass min_eps=0 to allow true 0% exploration (unlike EpsilonGreedyExploration)
        super().__init__(
            eps_init=explore_ratio,
            eps_final=explore_ratio,
            decay_episodes=1,  # Not used since eps_init == eps_final
            min_eps=0.0,  # Allow ratios from 0.0 to 1.0
        )
        # Store explore_ratio for direct access
        self.explore_ratio = explore_ratio

    def get_exploration_config(self) -> Dict[str, Any]:
        # Override to report explore_ratio for clarity
        return {"explore_ratio": self.explore_ratio}


class BasePolicyTrainer(ABC):
    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.env = env
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.replay_buffer = replay_buffer  # Placeholder for potential replay buffer
        self.eps = eps

        # Set up exploration module: default to pure exploit
        self.exploration_module = exploration_module or RandomExploration()

    @abstractmethod
    def fit(self, **kwargs):
        pass


class ReinforcePPTrainer(BasePolicyTrainer):
    """
    REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models.

    This class implements the REINFORCE++ algorithm which enhances the classical REINFORCE
    algorithm with PPO techniques but without a critic network. Key features:

    1. Mean reward baseline: Subtracts mean reward to reduce variance
    2. Reward normalization: Normalizes rewards across samples for stability
    3. PPO-style clipping: Uses clip-based policy gradient
    4. No critic network: Uses cumulative returns directly

    Reference: https://arxiv.org/abs/2501.03262
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer=None,
        clip_param: float = 0.2,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        use_reward_baseline: bool = True,
        use_reward_normalization: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        """
        Initialize REINFORCE++ optimizer.

        Args:
            policy_model: The policy model to optimize
            max_episodes: Number of episodes to train for
            optimizer: The optimizer to use for training
            lr_scheduler: Optional learning rate scheduler for dynamic learning rate
            exploration_module: Optional exploration exploitation module
            replay_buffer: Optional replay buffer
            clip_param: PPO-style clipping parameter
            num_mini_batch: Number of mini-batches per epoch
            entropy_coef: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_kl_loss: Whether to use KL loss from GRPO
            kl_coef: KL divergence coefficient
            use_reward_baseline: Whether to use mean reward baseline
            use_reward_normalization: Whether to normalize rewards
            gamma: Discount factor for return calculation (default 1.0 for REINFORCE++)
            eps: Epsilon for optimizer numerical stability
        """
        # TODO: Add support for replay buffer and exploration module in future iterations
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.use_reward_baseline = use_reward_baseline
        self.use_reward_normalization = use_reward_normalization
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

    def _compute_returns(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cumulative returns using REINFORCE approach (no GAE).

        Args:
            rewards: Rewards of shape (batch_size, sequence_length)
            masks: Binary masks indicating valid positions (batch_size, sequence_length)

        Returns:
            Returns of shape (batch_size, sequence_length)
        """
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros_like(rewards[:, 0])

        # Reverse time steps
        for t in reversed(range(rewards.size(1))):
            cumulative_return = (
                cumulative_return * self.gamma * masks[:, t] + rewards[:, t]
            )
            returns[:, t] = cumulative_return

        return returns

    def _compute_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using mean baseline (REINFORCE++ approach).

        Args:
            returns: Cumulative returns of shape (batch_size, sequence_length)

        Returns:
            Advantages of shape (batch_size, sequence_length)
        """
        # Flatten returns for normalization
        flat_returns = returns.flatten()

        # Compute mean baseline
        mean_return = flat_returns.mean()

        # Compute advantages
        advantages = returns - mean_return

        return advantages

    def _normalize_rewards(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize rewards across samples for stability.

        Args:
            rewards: Rewards of shape (batch_size, sequence_length)
            masks: Binary masks indicating valid positions

        Returns:
            Normalized rewards
        """
        # Apply mask
        masked_rewards = rewards * masks

        # Compute mean and std per sample
        batch_size = rewards.size(0)
        normalized_rewards = torch.zeros_like(rewards)

        for i in range(batch_size):
            sample_rewards = masked_rewards[i]
            valid_rewards = sample_rewards[sample_rewards != 0]
            if len(valid_rewards) > 1:
                mean = valid_rewards.mean()
                std = valid_rewards.std() + self.eps
                normalized_rewards[i] = (sample_rewards - mean) / std
            elif len(valid_rewards) == 1:
                normalized_rewards[i] = (
                    sample_rewards  # No normalization for single reward
                )
            else:
                normalized_rewards[i] = torch.zeros_like(sample_rewards)

        return normalized_rewards

    def update(
        self,
        obs,
        actions,
        rewards,
        masks,
        old_log_probs,
        kl_divergence=None,
    ):
        """
        Perform one update step of REINFORCE++.

        Args:
            obs: Observations
            actions: Actions taken
            rewards: Rewards received
            masks: Binary masks for valid positions
            old_log_probs: Previous log probabilities
            returns: Cumulative returns (optional, computed if not provided)
            advantages: Advantages (optional, computed if not provided)
            kl_divergence: KL divergence from reference policy (optional)

        Returns:
            Tuple with loss values
        """
        returns = self._compute_returns(rewards, masks)
        advantages = self._compute_advantages(returns)
        if self.use_reward_normalization:
            rewards = self._normalize_rewards(rewards, masks)

        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            # Get batch data
            obs_batch = obs[batch_indices]
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = (
                old_log_probs[batch_indices] if old_log_probs is not None else None
            )
            advantages_batch = advantages[batch_indices]

            # Forward pass
            _, action_log_probs, dist_entropy, _ = self.policy_model.evaluate_actions(
                obs_batch, actions_batch, masks_batch
            )

            # Compute policy loss with PPO-style clipping
            if old_log_probs_batch is not None:
                ratio = torch.exp(action_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * advantages_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()
            else:
                # If no old log probs, use unclipped policy gradient
                action_loss = -(action_log_probs * advantages_batch).mean()

            # Add entropy regularization
            action_loss = action_loss - self.entropy_coef * dist_entropy.mean()

            # Add KL loss if enabled
            if self.use_kl_loss and kl_divergence is not None:
                kl_loss = kl_divergence[batch_indices].mean()
                action_loss = action_loss + self.kl_coef * kl_loss
                kl_loss_epoch += kl_loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Step the learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = batch_size // self.num_mini_batch

        return (
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ) -> None:
        """Train the policy using REINFORCE++ or PPO algorithm.

        Args:
            max_episodes: Maximum number of training episodes
            checkpoint_every: Save checkpoint every N episodes
            earlystop_last: Early stop if no improvement in N episodes
            record_every: Record episode every N episodes
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: WandB project name
            checkpoint_dir: Directory to save checkpoints
            **kwargs: Additional arguments
        """
        # Initialize WandB if enabled
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[1].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        # Training loop
        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            # Reset environment
            obs, info = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
            }

            # Collect trajectory
            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                # Convert observation to tensor with proper shape for policy model
                # obs shape: (embedding_dim,) -> (1, 1, embedding_dim) for batch=1, seq=1
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )

                # Get action from policy with exploration
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    # Extract action from policy output using helper method
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    action = int(action_tensor.item())
                    log_prob = log_prob[0].item()

                # Take step in environment
                next_obs, reward, done, truncated, info = self.env.step(action)

                # Store transition
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)

                episode_reward += reward
                obs = next_obs
                step_count += 1

            # Convert to tensors
            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )

            # Update policy
            action_loss, dist_entropy, kl_loss = self.update(
                obs_tensor.unsqueeze(0),
                actions_tensor.unsqueeze(0),
                rewards_tensor.unsqueeze(0),
                masks_tensor.unsqueeze(0),
                old_log_probs_tensor.unsqueeze(0),
            )

            # Track metrics
            episode_rewards.append(episode_reward)

            # Check for improvement (early stopping)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Logging
            if use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "mean_reward_100": mean_reward,
                        "best_reward": best_reward,
                        "steps": step_count,
                        "action_loss": action_loss,
                        "dist_entropy": dist_entropy,
                        "kl_loss": kl_loss,
                    }
                )

            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )

            # Checkpointing
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            # Early stopping
            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class GRPOTrainer(BasePolicyTrainer):
    """Group Relative Policy Optimization trainer.

    This trainer implements a critic-free, group-relative advantage approach
    for prompt optimization. It uses groups of outputs sampled from the same
    prompt to compute relative advantages and optionally normalizes rewards
    within each group.

    References: https://arxiv.org/pdf/2402.03300
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer=None,
        clip_param: float = 0.2,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        group_size: int = 1,
        use_reward_normalization: bool = True,
        gamma: float = 1.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.group_size = max(1, group_size)
        self.use_reward_normalization = use_reward_normalization
        self.gamma = gamma

    def _compute_returns(
        self, rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros_like(rewards[:, 0])

        for t in reversed(range(rewards.size(1))):
            cumulative_return = (
                cumulative_return * self.gamma * masks[:, t] + rewards[:, t]
            )
            returns[:, t] = cumulative_return

        return returns

    def _compute_group_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        final_returns = returns[:, 0]
        batch_size = final_returns.size(0)

        if batch_size % self.group_size != 0:
            raise ValueError("GRPO requires batch_size to be divisible by group_size")

        group_returns = final_returns.view(-1, self.group_size)
        group_mean = group_returns.mean(dim=1, keepdim=True)

        if self.use_reward_normalization:
            group_std = group_returns.std(dim=1, unbiased=False, keepdim=True)
            normalized = (group_returns - group_mean) / (group_std + self.eps)
        else:
            normalized = group_returns - group_mean

        normalized_returns = normalized.view(batch_size)
        return normalized_returns.unsqueeze(-1).expand_as(returns)

    def update(
        self,
        obs,
        actions,
        rewards,
        masks,
        old_log_probs,
        kl_divergence=None,
    ):
        returns = self._compute_returns(rewards, masks)
        advantages = self._compute_group_advantages(returns)

        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            obs_batch = obs[batch_indices]
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = (
                old_log_probs[batch_indices] if old_log_probs is not None else None
            )
            advantages_batch = advantages[batch_indices]

            _, action_log_probs, dist_entropy, _ = self.policy_model.evaluate_actions(
                obs_batch, actions_batch, masks_batch
            )

            if old_log_probs_batch is not None:
                ratio = torch.exp(action_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    * advantages_batch
                )
                action_loss = -torch.min(surr1, surr2).mean()
            else:
                action_loss = -(action_log_probs * advantages_batch).mean()

            action_loss = action_loss - self.entropy_coef * dist_entropy.mean()

            if self.use_kl_loss and kl_divergence is not None:
                kl_loss = kl_divergence[batch_indices].mean()
                action_loss = action_loss + self.kl_coef * kl_loss
                kl_loss_epoch += kl_loss.item()

            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = batch_size // self.num_mini_batch
        return (
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ):
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_pattern = os.path.join(checkpoint_dir, "grpo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"grpo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            obs, _ = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
            }

            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    _, action_log_probs, _, _ = self.policy_model.evaluate_actions(
                        obs_tensor,
                        action_tensor,
                        torch.ones_like(action_tensor, dtype=torch.float32),
                    )

                action = int(action_tensor.item())
                log_prob = log_prob[0].item()

                next_obs, reward, done, truncated, _ = self.env.step(action)

                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)

                episode_reward += reward
                obs = next_obs
                step_count += 1

            if len(episode_data["observations"]) == 0:
                continue

            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )

            action_loss, dist_entropy, kl_loss = self.update(
                obs_tensor.unsqueeze(0),
                actions_tensor.unsqueeze(0),
                rewards_tensor.unsqueeze(0),
                masks_tensor.unsqueeze(0),
                old_log_probs_tensor.unsqueeze(0),
            )

            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "mean_reward_100": mean_reward,
                        "best_reward": best_reward,
                        "steps": step_count,
                        "action_loss": action_loss,
                        "dist_entropy": dist_entropy,
                        "kl_loss": kl_loss,
                    }
                )

            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )

            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"grpo_checkpoint_{episode}.pt"
                )
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class PPOTrainer(BasePolicyTrainer):
    def __init__(
        self,
        actor_critic: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer=None,
        clip_param=0.2,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss=True,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        gamma: float = 0.99,
        eps=1e-8,
        **kwargs,
    ):
        super().__init__(
            actor_critic,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.clip_param = clip_param
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.gamma = gamma

    def update(
        self,
        obs,
        actions,
        rewards,
        masks,
        old_log_probs,
        values,
        returns,
        old_logits,
    ) -> Tuple[float, float, float, float]:
        """Perform one update step of PPO.

        Args:
            obs: Observations
            actions: Actions taken
            rewards: Rewards received
            masks: Binary masks for valid positions
            old_log_probs: Previous log probabilities
            values: Previous value predictions
            returns: Computed returns
            old_logits: Previous logits for KL loss computation

        Returns:
            Tuple of (value_loss, action_loss, dist_entropy, kl_loss)
        """
        advantages = returns - values
        adv_std = advantages.std()
        if adv_std > self.eps:
            advantages = (advantages - advantages.mean()) / adv_std

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        num_updates = 0

        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]
            obs_batch = obs[batch_indices].unsqueeze(1)
            actions_batch = actions[batch_indices]
            masks_batch = masks[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            values_batch = values[batch_indices]
            return_batch = returns[batch_indices]
            adv_batch = advantages[batch_indices]

            values_pred, action_log_probs, dist_entropy, logits = (
                self.policy_model.evaluate_actions(
                    obs_batch,
                    actions_batch,
                    masks_batch,
                )
            )

            ratio = torch.exp(action_log_probs - old_log_probs_batch.unsqueeze(-1))
            surr1 = ratio * adv_batch.unsqueeze(-1)
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            ) * adv_batch.unsqueeze(-1)
            action_loss = -torch.min(surr1, surr2).mean()
            kl_loss = torch.tensor(0.0, device=logits.device)

            if self.use_kl_loss:
                old_logits_batch = old_logits[batch_indices]
                current_logits = logits.squeeze(1)
                old_probs = F.softmax(old_logits_batch, dim=-1)
                current_log_probs = F.log_softmax(current_logits, dim=-1)
                kl_divergence = F.kl_div(
                    current_log_probs,
                    old_probs,
                    reduction="none",
                ).sum(dim=-1)
                kl_loss = kl_divergence.mean()
                action_loss = action_loss + self.kl_coef * kl_loss

            if self.use_clipped_value_loss:
                value_pred_clipped = values_batch + (values_pred - values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values_pred - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (return_batch - values_pred).pow(2).mean()

            loss = (
                value_loss * self.value_loss_coef
                + action_loss
                - dist_entropy.mean() * self.entropy_coef
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.mean().item()
            if self.use_kl_loss:
                kl_loss_epoch += kl_loss.item()
            num_updates += 1

        return (
            value_loss_epoch / max(num_updates, 1),
            action_loss_epoch / max(num_updates, 1),
            dist_entropy_epoch / max(num_updates, 1),
            kl_loss_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        **kwargs,
    ):
        """Train the policy using PPO with on-policy environment interaction.

        Args:
            checkpoint_every: Save checkpoint every N episodes.
            earlystop_last: Early stop if no improvement in N episodes.
            record_every: Record episode every N episodes (not implemented).
            use_wandb: Whether to use Weights & Biases for logging.
            wandb_project: WandB project name.
            checkpoint_dir: Directory to save checkpoints.
            **kwargs: Additional arguments. Supported keys: gamma.
        """

        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "record_every": record_every,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "ppo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(
                        os.path.basename(f).split("_")[2].split(".")[0]
                    )  # ppo_checkpoint_{ep}.pt
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"ppo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                episode_start = latest_episode + 1
                best_reward = checkpoint.get("best_reward", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        episode_rewards = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            obs, _ = self.env.reset()
            self.exploration_module.reset(episode)
            episode_reward = 0.0
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "masks": [],
                "log_probs": [],
                "values": [],
                "old_logits": [],
            }

            done = False
            truncated = False
            step_count = 0

            while not (done or truncated) and step_count < self.env.max_steps:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                )
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Determine whether to explore or exploit
                    should_explore = self.exploration_module.should_explore(
                        step=step_count, episode=episode
                    )
                    action_tensor, log_prob = self.policy_model.get_action(
                        logits, deterministic=not should_explore
                    )
                    values, action_log_probs, _, _ = self.policy_model.evaluate_actions(
                        obs_tensor,
                        action_tensor,
                        torch.ones_like(action_tensor, dtype=torch.float32),
                    )

                action = int(action_tensor.item())
                log_prob = log_prob[0].item()
                value = values[0].item()

                next_obs, reward, done, truncated, _ = self.env.step(action)

                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["masks"].append(1.0 if not (done or truncated) else 0.0)
                episode_data["log_probs"].append(log_prob)
                episode_data["values"].append(value)
                episode_data["old_logits"].append(logits[0, 0].detach())

                episode_reward += reward
                obs = next_obs
                step_count += 1

            if len(episode_data["observations"]) == 0:
                continue

            obs_tensor = torch.tensor(episode_data["observations"], dtype=torch.float32)
            actions_tensor = torch.tensor(episode_data["actions"], dtype=torch.long)
            rewards_tensor = torch.tensor(episode_data["rewards"], dtype=torch.float32)
            masks_tensor = torch.tensor(episode_data["masks"], dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(
                episode_data["log_probs"], dtype=torch.float32
            )
            old_logits_tensor = torch.stack(episode_data["old_logits"], dim=0)
            values_tensor = torch.tensor(episode_data["values"], dtype=torch.float32)

            returns = torch.zeros_like(rewards_tensor)
            cumulative_return = torch.zeros_like(rewards_tensor[0])
            for t in reversed(range(rewards_tensor.size(0))):
                cumulative_return = (
                    cumulative_return * self.gamma * masks_tensor[t] + rewards_tensor[t]
                )
                returns[t] = cumulative_return

            value_loss, action_loss, dist_entropy, kl_loss = self.update(
                obs_tensor,
                actions_tensor,
                rewards_tensor,
                masks_tensor,
                old_log_probs_tensor,
                values_tensor,
                returns,
                old_logits_tensor,
            )

            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

            if mean_reward > best_reward:
                best_reward = mean_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if use_wandb:
                log_data = {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "mean_reward_100": mean_reward,
                    "best_reward": best_reward,
                    "steps": step_count,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "dist_entropy": dist_entropy,
                }
                if self.use_kl_loss:
                    log_data["kl_loss"] = kl_loss
                wandb.log(log_data)
            print(
                f"Episode {episode}: reward={episode_reward:.4f}, mean_100={mean_reward:.4f}"
            )

            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_checkpoint_{episode}.pt")
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": self.policy_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_reward": best_reward,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")


class DPOTrainer(BasePolicyTrainer):
    """Direct Preference Optimization (DPO) trainer for prompt augmentation.

    DPO is a parameter-efficient and computationally efficient approach to optimize
    prompts based on preference pairs. Unlike RLHF which requires a separate reward
    model, DPO directly optimizes the policy using preference data.

    Key differences from RLHF:
    - No explicit reward model training required
    - No critic network needed
    - Single-stage training on offline preference pairs
    - Implicit reward function derived from policy

    For prompt optimization:
    - We collect pairs of augmentations (preferred, rejected)
    - The preferred augmentation has higher reward (from reward_model)
    - We optimize the policy to prefer higher-reward augmentations

    Reference: https://arxiv.org/pdf/2305.18290

    Args:
        policy_model: The policy model to optimize
        env: PromptOptimizationEnv with reward_model and augmenters
        max_episodes: Maximum number of training episodes
        optimizer: PyTorch optimizer for parameter updates
        lr_scheduler: Optional learning rate scheduler
        exploration_module: Optional exploration module (unused for now)
        replay_buffer: Optional replay buffer (unused for now)
        beta: Temperature parameter controlling preference learning strength (default 0.1)
        num_mini_batch: Number of mini-batches per epoch (default 4)
        max_grad_norm: Maximum gradient norm for clipping (default 1.0)
        use_reference_policy: Whether to maintain separate reference policy (default True)
        eps: Epsilon for numerical stability (default 1e-8)
    """

    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module: Optional[BaseExplorationModule] = None,
        replay_buffer=None,
        beta: float = 0.1,
        num_mini_batch: int = 4,
        max_grad_norm: float = 1.0,
        use_reference_policy: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: The policy model to optimize
            env: PromptOptimizationEnv with reward_model for evaluating augmentations
            max_episodes: Maximum number of training episodes
            optimizer: PyTorch optimizer
            lr_scheduler: Optional learning rate scheduler
            exploration_module: Optional exploration module (not used currently)
            replay_buffer: Optional replay buffer (not used currently)
            beta: Temperature parameter (0.1-0.5 typical)
            num_mini_batch: Number of mini-batches for gradient updates
            max_grad_norm: Maximum gradient norm for clipping
            use_reference_policy: Whether to keep reference policy frozen
            eps: Numerical stability epsilon
        """
        super().__init__(
            policy_model,
            env,
            max_episodes,
            optimizer,
            lr_scheduler,
            exploration_module,
            replay_buffer,
            eps,
        )
        self.beta = beta
        self.num_mini_batch = num_mini_batch
        self.max_grad_norm = max_grad_norm
        self.use_reference_policy = use_reference_policy
        self.eps = eps

        # Create reference policy as a copy of the policy model if specified
        if self.use_reference_policy:
            self.reference_policy = type(policy_model)(
                policy_model.action_space,
                policy_model.observation_space,
            )
            # Copy state dict from policy model
            self.reference_policy.load_state_dict(policy_model.state_dict())
            # Freeze reference policy parameters
            for param in self.reference_policy.parameters():
                param.requires_grad = False
        else:
            self.reference_policy = None

    def _collect_preference_pairs(
        self, num_pairs: int = 32
    ) -> Tuple[list, list, list, list]:
        """Collect preference pairs by interacting with environment.

        For each base prompt, we try different augmentations and compare rewards
        to create preference pairs (preferred, rejected).

        Args:
            num_pairs: Number of preference pairs to collect

        Returns:
            Tuple of (preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards)
        """
        preferred_prompts = []
        rejected_prompts = []
        preferred_rewards = []
        rejected_rewards = []

        for _ in range(num_pairs):
            # Reset environment to get a starting prompt
            obs, info = self.env.reset()
            current_prompt = info["current_prompt"]

            # Collect augmentations and their rewards
            augmentation_rewards = []
            augmented_prompts = []

            for action_idx in range(len(self.env._augmenters)):
                augmenter = self.env._augmenters[action_idx]
                try:
                    # Apply augmentation
                    message = AgentMessage(query=current_prompt)
                    result_message = augmenter(message)
                    augmented_prompt = result_message.query

                    # Evaluate reward
                    reward = self.env._reward_model(augmented_prompt)

                    augmentation_rewards.append(reward)
                    augmented_prompts.append(augmented_prompt)
                except Exception:
                    # If augmentation fails, assign very low reward
                    augmentation_rewards.append(-float("inf"))
                    augmented_prompts.append(None)

            # Find preferred (highest reward) and rejected (lowest reward)
            valid_indices = [
                i for i, r in enumerate(augmentation_rewards) if r != -float("inf")
            ]

            if len(valid_indices) >= 2:
                # Sort by reward
                valid_indices_sorted = sorted(
                    valid_indices, key=lambda i: augmentation_rewards[i]
                )

                # Preferred: highest reward
                preferred_idx = valid_indices_sorted[-1]
                # Rejected: lowest reward (or second best if only 2)
                rejected_idx = valid_indices_sorted[0]

                preferred_prompts.append(augmented_prompts[preferred_idx])
                rejected_prompts.append(augmented_prompts[rejected_idx])
                preferred_rewards.append(augmentation_rewards[preferred_idx])
                rejected_rewards.append(augmentation_rewards[rejected_idx])

        return preferred_prompts, rejected_prompts, preferred_rewards, rejected_rewards

    def _compute_dpo_loss(
        self,
        policy_log_probs_w: torch.Tensor,
        policy_log_probs_l: torch.Tensor,
        reference_log_probs_w: torch.Tensor,
        reference_log_probs_l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss using the Bradley-Terry model.

        The DPO loss encourages the policy to increase the log probability of
        preferred augmentations while decreasing it for rejected ones.

        Loss = -log(sigma(β * (log_pi_w/pi_ref_w - log_pi_l/pi_ref_l)))

        Args:
            policy_log_probs_w: Log probs from policy for winning (preferred) actions
            policy_log_probs_l: Log probs from policy for losing (rejected) actions
            reference_log_probs_w: Log probs from reference for winning actions
            reference_log_probs_l: Log probs from reference for losing actions

        Returns:
            DPO loss (scalar)
        """
        # Compute log probability ratios (implicit rewards)
        # r_theta = beta * (log(π_theta(y_w|x)) - log(π_ref(y_w|x)))
        log_ratio_w = policy_log_probs_w - reference_log_probs_w
        log_ratio_l = policy_log_probs_l - reference_log_probs_l

        # DPO loss using sigmoid: -log(sigma(beta * (r_w - r_l)))
        log_ratios_diff = self.beta * (log_ratio_w - log_ratio_l)
        losses = -F.logsigmoid(log_ratios_diff)

        return losses.mean()

    def update(
        self,
        preferred_obs,
        rejected_obs,
        preferred_actions,
        rejected_actions,
        preferred_masks,
        rejected_masks,
    ) -> Tuple[float, float]:
        """Perform one update step using DPO loss.

        Args:
            preferred_obs: Observations for preferred prompts
            rejected_obs: Observations for rejected prompts
            preferred_actions: Actions taken for preferred prompts
            rejected_actions: Actions taken for rejected prompts
            preferred_masks: Masks for preferred prompts
            rejected_masks: Masks for rejected prompts

        Returns:
            Tuple of (dpo_loss, implicit_reward_diff)
        """
        dpo_loss_epoch = 0.0
        implicit_reward_diff_epoch = 0.0
        num_updates = 0

        batch_size = preferred_obs.size(0)
        indices = torch.randperm(batch_size)

        for i in range(0, batch_size, self.num_mini_batch):
            batch_indices = indices[i : i + self.num_mini_batch]

            # Get batch data
            pref_obs_batch = preferred_obs[batch_indices]
            rej_obs_batch = rejected_obs[batch_indices]
            pref_actions_batch = preferred_actions[batch_indices]
            rej_actions_batch = rejected_actions[batch_indices]
            pref_masks_batch = preferred_masks[batch_indices]
            rej_masks_batch = rejected_masks[batch_indices]

            # Forward pass for preferred actions through policy
            _, policy_log_probs_w, _, _ = self.policy_model.evaluate_actions(
                pref_obs_batch, pref_actions_batch, pref_masks_batch
            )

            # Forward pass for rejected actions through policy
            _, policy_log_probs_l, _, _ = self.policy_model.evaluate_actions(
                rej_obs_batch, rej_actions_batch, rej_masks_batch
            )

            # Get log probs from reference policy (no gradient)
            if self.use_reference_policy:
                with torch.no_grad():
                    _, reference_log_probs_w, _, _ = (
                        self.reference_policy.evaluate_actions(
                            pref_obs_batch, pref_actions_batch, pref_masks_batch
                        )
                    )
                    _, reference_log_probs_l, _, _ = (
                        self.reference_policy.evaluate_actions(
                            rej_obs_batch, rej_actions_batch, rej_masks_batch
                        )
                    )
            else:
                # If no reference policy, use detached policy logits
                with torch.no_grad():
                    _, reference_log_probs_w, _, _ = self.policy_model.evaluate_actions(
                        pref_obs_batch, pref_actions_batch, pref_masks_batch
                    )
                    _, reference_log_probs_l, _, _ = self.policy_model.evaluate_actions(
                        rej_obs_batch, rej_actions_batch, rej_masks_batch
                    )

            # Compute DPO loss
            dpo_loss = self._compute_dpo_loss(
                policy_log_probs_w,
                policy_log_probs_l,
                reference_log_probs_w,
                reference_log_probs_l,
            )

            # Compute implicit reward difference for logging
            with torch.no_grad():
                implicit_reward_w = self.beta * (
                    policy_log_probs_w - reference_log_probs_w
                )
                implicit_reward_l = self.beta * (
                    policy_log_probs_l - reference_log_probs_l
                )
                implicit_reward_diff = (implicit_reward_w - implicit_reward_l).mean()

            # Backward pass
            self.optimizer.zero_grad()
            dpo_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Step the learning rate scheduler if provided
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            dpo_loss_epoch += dpo_loss.item()
            implicit_reward_diff_epoch += implicit_reward_diff.item()
            num_updates += 1

        return (
            dpo_loss_epoch / max(num_updates, 1),
            implicit_reward_diff_epoch / max(num_updates, 1),
        )

    def fit(
        self,
        checkpoint_every: int = 10,
        earlystop_last: int = 100,
        record_every: int = 1000,
        use_wandb: bool = True,
        wandb_project: str = "prompt_optimization",
        checkpoint_dir: str = "./ckpt",
        pairs_per_episode: int = 32,
        **kwargs,
    ) -> None:
        """Train the policy using DPO on preference pairs.

        DPO training proceeds by:
        1. Collecting preference pairs through environment interaction
        2. Computing log probabilities under policy and reference
        3. Optimizing using the DPO loss (Bradley-Terry model)

        Args:
            checkpoint_every: Save checkpoint every N episodes
            earlystop_last: Early stop if no improvement in N episodes
            record_every: Record metrics every N episodes
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: WandB project name
            checkpoint_dir: Directory to save checkpoints
            pairs_per_episode: Number of preference pairs to collect per episode
            **kwargs: Additional arguments
        """
        # Initialize WandB if enabled
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "max_episodes": self.max_episodes,
                        "beta": self.beta,
                        "checkpoint_every": checkpoint_every,
                        "earlystop_last": earlystop_last,
                        "pairs_per_episode": pairs_per_episode,
                    },
                )
            except ImportError:
                warnings.warn("wandb not installed, skipping WandB logging")
                use_wandb = False

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load latest checkpoint if exists
        ckpt_pattern = os.path.join(checkpoint_dir, "dpo_checkpoint_*.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        episode_start = 0
        best_reward_diff = -float("inf")
        if ckpt_files:
            episodes = []
            for f in ckpt_files:
                try:
                    ep = int(os.path.basename(f).split("_")[2].split(".")[0])
                    episodes.append(ep)
                except ValueError:
                    pass
            if episodes:
                latest_episode = max(episodes)
                ckpt_path = os.path.join(
                    checkpoint_dir, f"dpo_checkpoint_{latest_episode}.pt"
                )
                checkpoint = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.use_reference_policy:
                    self.reference_policy.load_state_dict(
                        checkpoint["reference_policy_state_dict"]
                    )
                episode_start = latest_episode + 1
                best_reward_diff = checkpoint.get("best_reward_diff", -float("inf"))
                print(f"Loaded checkpoint from episode {latest_episode}")

        # Training loop
        episode_reward_diffs = []
        no_improvement_count = 0

        for episode in range(episode_start, self.max_episodes):
            # Collect preference pairs
            (
                preferred_prompts,
                rejected_prompts,
                preferred_rewards,
                rejected_rewards,
            ) = self._collect_preference_pairs(num_pairs=pairs_per_episode)

            if len(preferred_prompts) == 0:
                print(f"Episode {episode}: No valid preference pairs collected")
                continue

            # Convert prompts to observations using embedding model
            preferred_obs_list = []
            rejected_obs_list = []

            for pref_prompt, rej_prompt in zip(preferred_prompts, rejected_prompts):
                pref_embedding = self.env._embedding_model(pref_prompt).astype(
                    np.float32
                )
                rej_embedding = self.env._embedding_model(rej_prompt).astype(np.float32)
                preferred_obs_list.append(pref_embedding)
                rejected_obs_list.append(rej_embedding)

            # For simplicity, use action 0 (can be extended to track actual actions)
            batch_size = len(preferred_prompts)
            preferred_obs = torch.tensor(
                np.stack(preferred_obs_list, axis=0), dtype=torch.float32
            )
            rejected_obs = torch.tensor(
                np.stack(rejected_obs_list, axis=0), dtype=torch.float32
            )

            # Add sequence dimension for compatibility with policy model
            preferred_obs = preferred_obs.unsqueeze(1)  # (batch, 1, obs_dim)
            rejected_obs = rejected_obs.unsqueeze(1)  # (batch, 1, obs_dim)

            # For prompt augmentation, the "action" is implicit in the augmentation
            # We use action 0 as a placeholder
            preferred_actions = torch.zeros(batch_size, 1, dtype=torch.long)
            rejected_actions = torch.zeros(batch_size, 1, dtype=torch.long)
            preferred_masks = torch.ones(batch_size, 1, dtype=torch.float32)
            rejected_masks = torch.ones(batch_size, 1, dtype=torch.float32)

            # Update policy using DPO loss
            dpo_loss, reward_diff = self.update(
                preferred_obs,
                rejected_obs,
                preferred_actions,
                rejected_actions,
                preferred_masks,
                rejected_masks,
            )

            # Track metrics
            episode_reward_diffs.append(reward_diff)

            # Check for improvement (early stopping)
            mean_reward_diff = np.mean(
                episode_reward_diffs[-min(100, len(episode_reward_diffs)) :]
            )

            if mean_reward_diff > best_reward_diff:
                best_reward_diff = mean_reward_diff
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Logging
            if use_wandb:
                mean_preferred_reward = float(np.mean(preferred_rewards))
                mean_rejected_reward = float(np.mean(rejected_rewards))
                wandb.log(
                    {
                        "episode": episode,
                        "dpo_loss": dpo_loss,
                        "reward_diff": reward_diff,
                        "mean_reward_diff_100": mean_reward_diff,
                        "best_reward_diff": best_reward_diff,
                        "mean_preferred_reward": mean_preferred_reward,
                        "mean_rejected_reward": mean_rejected_reward,
                        "pairs_collected": len(preferred_prompts),
                    }
                )

            print(
                f"Episode {episode}: loss={dpo_loss:.4f}, reward_diff={reward_diff:.4f}, "
                f"mean_diff_100={mean_reward_diff:.4f}"
            )

            # Checkpointing
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"dpo_checkpoint_{episode}.pt")
                checkpoint_dict = {
                    "episode": episode,
                    "model_state_dict": self.policy_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_reward_diff": best_reward_diff,
                }
                if self.use_reference_policy:
                    checkpoint_dict["reference_policy_state_dict"] = (
                        self.reference_policy.state_dict()
                    )

                torch.save(checkpoint_dict, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # Early stopping
            if no_improvement_count >= earlystop_last:
                print(f"Early stopping at episode {episode}")
                break

        if use_wandb:
            wandb.finish()
        print("done")
