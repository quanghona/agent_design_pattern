import gymnasium
import numpy as np
import torch
from aap_core.policy import GPT2Policy


def test_gpt2_policy_save_load(tmp_path):
    # Setup environment parameters
    action_space = gymnasium.spaces.Discrete(10)
    observation_space = gymnasium.spaces.Box(
        low=-1.0, high=1.0, shape=(768,), dtype=np.float32
    )

    # Initialize policy
    policy = GPT2Policy(
        action_space=action_space,
        observation_space=observation_space,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
    )
    policy.eval()

    # Create dummy input
    batch_size = 4
    obs = torch.randn(batch_size, 768)

    # Get original output
    with torch.no_grad():
        original_logits = policy(obs)

    # Save policy
    save_path = tmp_path / "test_policy.pth"
    print(f"Saving policy to {save_path}")
    policy.save(str(save_path))

    # Create a new policy instance with same configuration
    new_policy = GPT2Policy(
        action_space=action_space,
        observation_space=observation_space,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
    )
    new_policy.eval()

    # Load policy
    print(f"Loading policy from {save_path}")
    new_policy.load(str(save_path))

    # Get new output
    with torch.no_grad():
        loaded_logits = new_policy(obs)

    # Compare outputs
    diff = torch.abs(original_logits - loaded_logits).max().item()
    print(f"Max difference between original and loaded logits: {diff}")

    assert diff < 1e-6, f"Difference too large: {diff}"
