import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Check if the task is completed
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    if current_qpos > target_qpos:
        # Large reward for task completion
        reward += 1.0

    # Stage 2: Penalize excessive actions to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward