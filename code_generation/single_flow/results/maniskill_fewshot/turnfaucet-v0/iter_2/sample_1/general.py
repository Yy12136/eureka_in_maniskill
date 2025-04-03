import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get the current qpos of the faucet handle
    current_qpos = self.faucet.handle.qpos
    target_qpos = self.faucet.handle.target_qpos

    # Check if the task is completed
    if current_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Grasp the faucet handle
    is_grasped = self.robot.check_grasp(self.faucet.handle, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # Stage 2: Rotate the faucet handle
    if is_grasped:
        # Calculate the progress in rotating the handle
        rotation_progress = current_qpos / target_qpos
        if rotation_progress > 0.5:  # Reward for significant progress
            reward += 3.0
        elif rotation_progress > 0.2:  # Reward for initial progress
            reward += 1.0

    return reward