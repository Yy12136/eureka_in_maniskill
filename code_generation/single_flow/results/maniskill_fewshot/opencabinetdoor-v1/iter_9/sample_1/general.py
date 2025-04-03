import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Grasping the handle
    # Calculate the distance between the end-effector and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle

    # Check if the gripper is close enough to the handle to consider it grasped
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Stage 2: Opening the door
        # Reward based on how much the door has been opened
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        if door_progress >= 1.0:
            reward += 8.0  # Reward for fully opening the door
        elif door_progress > 0.5:
            reward += 3.0  # Milestone reward for significant progress
        elif door_progress > 0.1:
            reward += 1.0  # Milestone reward for initial progress

    return reward