from aap_core.policy import BasePolicy
import torch

import torch.nn as nn
import torch.optim as optim


class ReinforcePP:
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
        clip_param: float = 0.2,
        reinforce_epoch: int = 10,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        lr: float = 1e-6,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        use_kl_loss: bool = False,
        kl_coef: float = 1e-5,
        use_reward_baseline: bool = True,
        use_reward_normalization: bool = True,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        **kwargs,
    ):
        """
        Initialize REINFORCE++ optimizer.

        Args:
            policy_model: The policy model to optimize (should have evaluate_actions method)
            clip_param: PPO-style clipping parameter
            reinforce_epoch: Number of optimization epochs
            num_mini_batch: Number of mini-batches per epoch
            entropy_coef: Entropy regularization coefficient
            lr: Learning rate (used if lr_scheduler is None)
            eps: Epsilon for optimizer numerical stability
            max_grad_norm: Maximum gradient norm for clipping
            use_kl_loss: Whether to use KL loss from GRPO
            kl_coef: KL divergence coefficient
            use_reward_baseline: Whether to use mean reward baseline
            use_reward_normalization: Whether to normalize rewards
            lr_scheduler: Optional learning rate scheduler for dynamic learning rate
        """
        # Register the policy model as a submodule
        self.policy_model = policy_model
        self.clip_param = clip_param
        self.reinforce_epoch = reinforce_epoch
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.use_kl_loss = use_kl_loss
        self.kl_coef = kl_coef
        self.use_reward_baseline = use_reward_baseline
        self.use_reward_normalization = use_reward_normalization
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler = lr_scheduler

        # Create optimizer with the policy model's parameters
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr, eps=eps)

    def __getattr__(self, name: str):
        """Delegate attribute access to the policy model if not found in self."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # Delegate to the wrapped policy model
            if hasattr(self.policy_model, name):
                return getattr(self.policy_model, name)
            raise

    def _compute_returns(
        self, rewards: torch.Tensor, masks: torch.Tensor, gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute cumulative returns using REINFORCE approach (no GAE).

        Args:
            rewards: Rewards of shape (batch_size, sequence_length)
            masks: Binary masks indicating valid positions (batch_size, sequence_length)
            gamma: Discount factor (default 1.0 for REINFORCE++)

        Returns:
            Returns of shape (batch_size, sequence_length)
        """
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros_like(rewards[:, 0])

        # Reverse time steps
        for t in reversed(range(rewards.size(1))):
            cumulative_return = cumulative_return * gamma * masks[:, t] + rewards[:, t]
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

    def update(self, batch):
        """
        Perform one update step of REINFORCE++.

        Args:
            batch: Dictionary containing:
                - obs: Observations
                - actions: Actions taken
                - rewards: Rewards received
                - masks: Binary masks for valid positions
                - old_log_probs: Previous log probabilities
                - values: Value predictions (not used in REINFORCE++)
                - returns: Cumulative returns (optional, computed if not provided)
                - advantages: Advantages (optional, computed if not provided)
                - kl_divergence: KL divergence from reference policy (optional)

        Returns:
            Dictionary with loss values
        """
        obs = batch.get("obs")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        masks = batch.get("masks")
        old_log_probs = batch.get("old_log_probs")
        kl_divergence = batch.get("kl_divergence")

        # Compute returns if not provided
        if "returns" not in batch:
            returns = self._compute_returns(rewards, masks)
        else:
            returns = batch["returns"]

        # Compute advantages if not provided
        if "advantages" not in batch:
            advantages = self._compute_advantages(returns)
        else:
            advantages = batch["advantages"]

        # Normalize rewards if enabled
        if self.use_reward_normalization:
            rewards = self._normalize_rewards(rewards, masks)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        kl_loss_epoch = 0.0

        # Create data batches
        batch_size = rewards.size(0)
        indices = torch.randperm(batch_size)

        for e in range(self.reinforce_epoch):
            for i in range(0, batch_size, self.num_mini_batch):
                batch_indices = indices[i : i + self.num_mini_batch]

                # Get batch data
                obs_batch = obs[batch_indices] if obs is not None else None
                actions_batch = actions[batch_indices]
                masks_batch = masks[batch_indices]
                old_log_probs_batch = (
                    old_log_probs[batch_indices] if old_log_probs is not None else None
                )
                advantages_batch = advantages[batch_indices]

                # Forward pass
                values, action_log_probs, dist_entropy, _ = (
                    self.policy_model.evaluate_actions(
                        obs_batch, actions_batch, masks_batch
                    )
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

                # No value loss in REINFORCE++ (no critic network)
                value_loss = torch.tensor(0.0, device=rewards.device)

                # Backward pass
                self.optimizer.zero_grad()
                (value_loss + action_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Step the learning rate scheduler if provided
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = self.reinforce_epoch * (batch_size // self.num_mini_batch)

        return {
            "value_loss": value_loss_epoch / max(num_updates, 1),
            "action_loss": action_loss_epoch / max(num_updates, 1),
            "entropy": dist_entropy_epoch / max(num_updates, 1),
            "kl_loss": kl_loss_epoch / max(num_updates, 1),
        }


class PPO:
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.eps = eps

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = (
                    self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        masks_batch,
                        actions_batch,
                    )
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


# def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta=0.1):
#     """
#     pi_logps: policy logprobs, shape (B,)
#     ref_logps: reference model logprobs, shape (B,)
#     yw_idxs: preferred completion indices in [0, B-1], shape (T,)
#     yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
#     beta: temperature controlling strength of KL penalty
#     Each pair of (yw_idxs[i], yl_idxs[i]) represents the
#     indices of a single preference pair.
#     """
#     pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
#     ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
#     pi_logratios = pi_yw_logps - pi_yl_logps
#     ref_logratios = ref_yw_logps - ref_yl_logps
#     losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
#     rewards = beta * (pi_logps - ref_logps).detach()
#     return losses, rewards
