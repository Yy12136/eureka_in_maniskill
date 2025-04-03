import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Grasp the handle
    # Check if the handle is grasped
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    if is_grasped:
        # Reward for successful grasp
        reward += 0.5

        # Stage 2: Rotate the handle
        current_qpos = self.current_angle
        target_qpos = self.target_angle

        # Reward for rotating the handle towards the target
        rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)
        reward += 0.3 * rotation_progress

        # Stage 3: Task completion
        if current_qpos > target_qpos:
            # Large reward for task completion
            reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward