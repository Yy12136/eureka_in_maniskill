import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Check if the handle is grasped
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    if is_grasped:
        # Reward for successful grasp
        reward += 0.3

        # Stage 2: Check if the handle is rotated beyond the target angle
        current_qpos = self.current_angle
        target_qpos = self.target_angle
        if current_qpos > target_qpos:
            # Large reward for task completion
            reward += 1.0

    # Stage 3: Penalize excessive action magnitude to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward