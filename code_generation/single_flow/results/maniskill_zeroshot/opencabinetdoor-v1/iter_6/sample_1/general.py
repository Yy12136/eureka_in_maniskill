import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Robot reaches the cabinet
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_position = self.agent.base_pose.p[:2]
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
    if base_to_cabinet_dist < 0.1:  # Threshold for reaching the cabinet
        reward += 0.2  # Sparse reward for reaching the cabinet

    # Milestone 2: Robot grasps the handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:  # Threshold for grasping the handle
        reward += 0.3  # Sparse reward for grasping the handle

    # Milestone 3: Robot opens the door to the target position
    if cabinet_qpos >= target_qpos:  # Task completion condition
        reward += 1.0  # Sparse reward for completing the task

    # Penalty for task failure
    # If the robot moves too far from the cabinet or fails to grasp the handle, penalize
    if base_to_cabinet_dist > 1.0:  # Threshold for being too far from the cabinet
        reward -= 0.5  # Penalty for moving too far away
    if ee_to_handle_dist > 0.1:  # Threshold for failing to grasp the handle
        reward -= 0.3  # Penalty for failing to grasp

    return reward