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
        return reward  # Early return since the task is done

    # Stage 2: Sparse guidance for rotation progress
    # Only provide a small reward if the handle is being rotated towards the target
    if self.agent.check_grasp(self.target_link, max_angle=30):  # Check if the handle is grasped
        rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)
        reward += 0.1 * rotation_progress  # Small reward for progress

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward