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
        return reward  # Terminate early if task is completed

    # Milestone 2: Door opening progress
    # Only reward progress if the door is being opened
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    if door_progress > 0:
        reward += 0.5  # Moderate reward for making progress

    # Milestone 3: Grasping the handle
    # Only reward grasping if the door is not yet opened
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:
        reward += 0.3  # Small reward for grasping the handle

    # Milestone 4: Approaching the cabinet
    # Only reward approaching if the robot is not yet grasping the handle
    if ee_to_handle_dist >= 0.05:
        base_position = self.agent.base_pose.p[:2]
        cabinet_position = self.cabinet.pose.p[:2]
        base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
        if base_to_cabinet_dist < 0.1:
            reward += 0.1  # Small reward for approaching the cabinet

    return reward