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
        return reward  # Early termination if task is completed

    # Milestone 2: Door opening progress
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    reward += 0.1 * door_progress  # Small reward for door opening progress

    # Milestone 3: Distance between end-effector and cabinet handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:  # If gripper is close to handle
        reward += 0.05  # Small reward for being near the handle

    # Milestone 4: Gripper openness
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    if ee_to_handle_dist < 0.05 and abs(gripper_openness) < 0.1:  # If gripper is closed near the handle
        reward += 0.05  # Small reward for correct gripper state

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward