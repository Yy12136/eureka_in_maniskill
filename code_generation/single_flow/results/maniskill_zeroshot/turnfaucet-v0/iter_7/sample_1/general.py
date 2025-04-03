import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Task completion
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    # Check if the task is completed
    if current_qpos > target_qpos:
        # Large reward for task completion
        reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward