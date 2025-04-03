import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 100.0  # Large reward for task completion
        return reward

    # Stage 1: Approaching the cabinet handle
    # Calculate the distance between the end-effector and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle

    # Milestone: Reward for being close enough to attempt grasping
    if dist_to_handle < 0.05:
        reward += 10.0  # Milestone reward for being close to the handle

    # Stage 2: Grasping the handle
    # Check if the gripper is close enough to the handle to consider it grasped
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 20.0  # Reward for successful grasp

        # Stage 3: Opening the door
        # Reward based on how much the door has been opened
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        opening_reward = np.clip(door_progress, 0.0, 1.0)  # Reward proportional to door opening progress
        reward += 30.0 * opening_reward

    # Penalize large actions to encourage smooth movements
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty

    # Penalize unnecessary mobile base movement
    base_velocity_penalty = -0.1 * np.linalg.norm(self.robot.base_velocity)
    reward += base_velocity_penalty

    return reward