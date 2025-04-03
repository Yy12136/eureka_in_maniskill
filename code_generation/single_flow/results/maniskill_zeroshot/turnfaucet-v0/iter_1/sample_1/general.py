import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Check if the handle is grasped
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    if not is_grasped:
        # No reward until the handle is grasped
        return reward

    # Stage 2: Rotate the handle towards the target
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    if current_qpos >= target_qpos:
        # Large reward for task completion
        reward += 1.0
    else:
        # Small reward for making progress towards the target
        rotation_progress = max(0, (current_qpos - self.initial_angle) / (target_qpos - self.initial_angle))
        reward += 0.1 * rotation_progress

    return reward