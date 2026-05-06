from abc import ABC, abstractmethod
import os
import warnings

from aap_core.policy import BasePolicy
from aap_core.prompt_augmenter import PromptOptimizationEnv
import numpy as np
import torch

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
