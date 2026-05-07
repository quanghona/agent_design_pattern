from abc import ABC, abstractmethod
import os
import warnings

from aap_core.policy import BasePolicy
from aap_core.prompt_augmenter import PromptOptimizationEnv
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
import wandb


class BasePolicyTrainer(ABC):
    def __init__(
        self,
        policy_model: BasePolicy,
        env: PromptOptimizationEnv,
        max_episodes: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        exploration_module=None,
        replay_buffer=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.env = env
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.exploration_module = exploration_module
        self.replay_buffer = replay_buffer  # Placeholder for potential replay buffer
        self.eps = eps

    @abstractmethod
    def fit(self, **kwargs):
        pass


class ReinforcePP(BasePolicyTrainer):
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
        exploration_module=None,
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

        # Training loop
        episode_rewards = []
        best_reward = -float("inf")
        no_improvement_count = 0

        for episode in range(self.max_episodes):
            # Reset environment
            obs, info = self.env.reset()
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

                # Get action from policy
                with torch.no_grad():
                    logits = self.policy_model(obs_tensor)
                    # Extract action from policy output using helper method
                    action_tensor, _ = self.policy_model.get_action(logits)
                    action = int(action_tensor.item())

                    # For logging: compute log prob of the selected action
                    last_logits = logits[:, -1, :]
                    log_probs_dist = torch.nn.functional.log_softmax(
                        last_logits, dim=-1
                    )
                    log_prob = log_probs_dist[0, action].item()

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


class PPO(BasePolicyTrainer):
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
        exploration_module=None,
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
        episode_rewards = []
        best_reward = -float("inf")
        no_improvement_count = 0
        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0
        num_updates = 0

        for episode in range(self.max_episodes):
            obs, _ = self.env.reset()
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
                    action_tensor, _ = self.policy_model.get_action(logits)
                    values, action_log_probs, _, _ = self.policy_model.evaluate_actions(
                        obs_tensor,
                        action_tensor,
                        torch.ones_like(action_tensor, dtype=torch.float32),
                    )

                action = int(action_tensor.item())
                log_prob = action_log_probs[0].item()
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

            advantages = returns - values_tensor
            adv_std = advantages.std()
            if adv_std > self.eps:
                advantages = (advantages - advantages.mean()) / adv_std

            batch_size = rewards_tensor.size(0)
            indices = torch.randperm(batch_size)

            for i in range(0, batch_size, self.num_mini_batch):
                batch_indices = indices[i : i + self.num_mini_batch]
                obs_batch = obs_tensor[batch_indices].unsqueeze(1)
                actions_batch = actions_tensor[batch_indices]
                masks_batch = masks_tensor[batch_indices]
                old_log_probs_batch = old_log_probs_tensor[batch_indices]
                values_batch = values_tensor[batch_indices]
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
                    old_logits_batch = old_logits_tensor[batch_indices]
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
                    value_pred_clipped = values_batch + (
                        values_pred - values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values_pred - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values_pred).pow(2).mean()

                loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy.mean() * self.entropy_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.mean().item()
                if self.use_kl_loss:
                    kl_loss_epoch += kl_loss.item()
                num_updates += 1

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
                    "value_loss": value_loss_epoch / max(num_updates, 1),
                    "action_loss": action_loss_epoch / max(num_updates, 1),
                    "dist_entropy": dist_entropy_epoch / max(num_updates, 1),
                }
                if self.use_kl_loss:
                    log_data["kl_loss"] = kl_loss_epoch / max(num_updates, 1)
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

