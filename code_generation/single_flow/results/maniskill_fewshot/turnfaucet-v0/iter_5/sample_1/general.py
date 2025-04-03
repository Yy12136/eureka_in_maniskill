import numpy as np

def compute_sparse_reward(self) -> float:
    reward = 0.0

    # Get the current qpos of the faucet handle
    current_qpos = self.faucet.handle.qpos
    target_qpos = self.faucet.handle.target_qpos

    # Check if the task is completed
    if current_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    else:
        reward += 0.0  # No reward until the task is completed

    return reward