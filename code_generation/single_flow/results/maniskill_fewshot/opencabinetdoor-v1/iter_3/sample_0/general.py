import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 100.0  # Large reward for task completion
        return reward

    # Stage 1: Grasping the handle
    # Calculate the distance between the end-effector and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()
    handle_pcd = self.cabinet.handle.get_world_pcd()
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()

    # Reward for successful grasp
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 10.0  # Milestone reward for grasping the handle

    # Stage 2: Opening the door
    # Reward for making progress in opening the door
    door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
    if door_progress > 0.0:
        reward += 5.0 * door_progress  # Reward proportional to door opening progress

    return reward