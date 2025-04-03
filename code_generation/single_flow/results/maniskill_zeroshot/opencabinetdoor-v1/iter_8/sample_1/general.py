import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large reward for task completion

    # Milestone 2: Door opening progress (optional, for intermediate feedback)
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    reward += 0.1 * door_progress  # Small reward for progress

    return reward