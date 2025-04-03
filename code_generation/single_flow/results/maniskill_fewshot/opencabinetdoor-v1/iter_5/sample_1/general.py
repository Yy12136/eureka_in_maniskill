import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Task completion: Large reward for opening the door
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 10.0  # Sparse reward for task completion
        return reward

    # Milestone 1: Approaching the cabinet handle
    # Reward for getting close to the handle
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle
    if dist_to_handle < 0.05:  # Threshold for being close to the handle
        reward += 1.0  # Sparse milestone reward

    # Milestone 2: Grasping the handle
    # Reward for successfully grasping the handle
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 2.0  # Sparse milestone reward

    # Milestone 3: Progress in opening the door
    # Reward proportional to the door opening progress
    if is_grasped:
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        reward += 3.0 * door_progress  # Sparse reward proportional to progress

    return reward