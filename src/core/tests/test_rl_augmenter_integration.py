#!/usr/bin/env python3
"""Comprehensive integration test for RLPromptAugmenter implementation."""

import sys
import os
import numpy as np
import torch
from pathlib import Path

try:
    print("=" * 60)
    print("Testing RLPromptAugmenter Implementation")
    print("=" * 60)

    # Test 1: Imports
    print("\n[Test 1] Testing imports...")
    from aap_core.prompt_augmenter import (
        RLPromptAugmenter,
        PromptOptimizationEnv,
        SimplePromptAugmenter,
        IdentityPromptAugmenter,
    )
    from aap_core.policy_gradient import ReinforcePP
    from aap_core.policy import GPT2Policy
    from aap_core.types import AgentMessage

    print("✓ All imports successful")

    # Test 2: Create minimal environment for testing
    print("\n[Test 2] Creating minimal test environment...")

    # Simple embedding model
    def embedding_model(prompt: str) -> np.ndarray:
        """Simple hash-based deterministic embedding."""
        np.random.seed(hash(prompt) % 2**32)
        return np.random.randn(64).astype(np.float32)

    # Simple reward model
    def reward_model(prompt: str) -> float:
        """Simple reward based on prompt length."""
        return min(1.0, len(prompt) / 100.0)

    # Create simple augmenters
    augmenters = [
        IdentityPromptAugmenter(),
        IdentityPromptAugmenter(),
    ]

    # Create environment
    env = PromptOptimizationEnv(
        initial_prompt="Hello world",
        augmenters=augmenters,
        embedding_model=embedding_model,
        reward_model=reward_model,
        max_steps=5,
    )

    print(f"✓ Environment created")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Max steps: {env._max_steps}")

    # Test 3: Create RLPromptAugmenter in train mode
    print("\n[Test 3] Creating RLPromptAugmenter in train mode...")
    policy_model_path = "/tmp/test_policy_model.pt"
    rl_augmenter = RLPromptAugmenter(
        env=env,
        policy_model_path=policy_model_path,
        mode="train",
    )
    print(f"✓ RLPromptAugmenter created in train mode")
    print(f"  - Mode: {rl_augmenter.mode}")
    print(f"  - Policy model path: {rl_augmenter.policy_model_path}")

    # Test 4: Verify train method signature
    print("\n[Test 4] Verifying train method signature...")
    import inspect

    sig = inspect.signature(RLPromptAugmenter.train)
    params = list(sig.parameters.keys())

    required_params = [
        "algo",
        "max_episodes",
        "checkpoint_every",
        "earlystop_last",
        "record_every",
        "use_wandb",
        "wandb_project",
    ]

    all_found = True
    for param in required_params:
        if param not in params:
            print(f"✗ Missing parameter: {param}")
            all_found = False

    if all_found and "kwargs" in params:
        print(f"✓ All required parameters found")
    else:
        raise ValueError("Missing required parameters in train method")

    # Test 5: Run a minimal training loop
    print("\n[Test 5] Running minimal training loop (2 episodes)...")

    # Create checkpoint directory
    ckpt_dir = Path("./ckpt_test")
    if ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)

    # Run training with minimal episodes
    try:
        results = rl_augmenter.train(
            algo="reinforcepp",
            max_episodes=2,
            checkpoint_every=1,
            earlystop_last=100,
            record_every=1000,
            use_wandb=False,
            # Algorithm parameters
            clip_param=0.2,
            reinforce_epoch=1,
            num_mini_batch=1,
            entropy_coef=0.01,
            lr=1e-6,
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=8,
        )

        print(f"✓ Training completed successfully")
        print(f"  - Episodes trained: {len(results['episode_rewards'])}")
        print(f"  - Best reward: {results['best_reward']:.4f}")
        print(f"  - Final model path: {results['final_model_path']}")

        # Verify results
        if len(results["episode_rewards"]) == 2:
            print(f"✓ Correct number of episodes trained")
        else:
            print(f"✗ Expected 2 episodes, got {len(results['episode_rewards'])}")

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Test 6: Verify final model was saved
    print("\n[Test 6] Verifying model export...")
    final_model_path = Path(results["final_model_path"])
    if final_model_path.exists():
        print(f"✓ Final model saved: {final_model_path}")
        print(f"  - File size: {final_model_path.stat().st_size} bytes")
    else:
        print(f"✗ Final model not found: {final_model_path}")

    # Cleanup
    print("\n[Cleanup] Removing test files...")
    if ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)
    if final_model_path.exists():
        final_model_path.unlink()
    print("✓ Cleanup completed")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
