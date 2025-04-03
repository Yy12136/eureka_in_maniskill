import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Grasp the faucet handle
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    if is_grasped:
        # Reward for successful grasp
        reward += 0.5

        # Stage 2: Rotate the handle to the target position
        current_qpos = self.current_angle
        target_qpos = self.target_angle

        # Check if the task is completed
        if current_qpos > target_qpos:
            # Large reward for task completion
            reward += 1.0
        else:
            # Small reward for progress towards the target
            rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)
            reward += 0.1 * rotation_progress

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward