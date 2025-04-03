import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_qpos = self.target_link.qpos
    target_qpos = self.target_link.target_qpos

    # Milestone 1: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large reward for task completion

    # Regularization: Penalize large actions to encourage efficiency
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward