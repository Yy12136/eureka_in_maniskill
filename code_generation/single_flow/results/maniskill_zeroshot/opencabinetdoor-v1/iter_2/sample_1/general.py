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

    # Regularization: Penalize large actions (optional)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward