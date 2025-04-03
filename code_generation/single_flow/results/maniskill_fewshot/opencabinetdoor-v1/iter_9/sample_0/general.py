import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 10.0  # Large reward for task completion
        return reward

    # Milestone 1: Reward for being close enough to attempt grasping
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle
    if dist_to_handle < 0.05:
        reward += 1.0  # Milestone reward for being close to the handle

    # Milestone 2: Reward for successful grasp
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # Milestone 3: Reward for significant door opening progress
    door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
    if door_progress >= 0.5:  # Reward when the door is halfway open
        reward += 3.0

    return reward