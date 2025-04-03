import numpy as np

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
        return reward  # Early return to focus on task completion

    # Stage 2: Grasp the handle (optional, for intermediate progress)
    if self.agent.check_grasp(self.target_link, max_angle=30):
        # Small reward for successful grasp
        reward += 0.1

    # Stage 3: Rotate the handle (optional, for intermediate progress)
    if current_qpos > 0:  # If the handle has been rotated at all
        # Small reward for any rotation progress
        reward += 0.1

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward