import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large reward for task completion
        return reward  # Return immediately upon task completion

    # Milestone 2: Distance between end-effector and cabinet handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:  # If gripper is close to handle
        reward += 0.1  # Small reward for being near the handle

    # Milestone 3: Door opening progress
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    if door_progress > 0:
        reward += 0.2  # Medium reward for starting to open the door

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward