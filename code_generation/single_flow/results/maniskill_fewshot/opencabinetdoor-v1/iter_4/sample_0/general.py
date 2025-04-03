import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Approaching the cabinet handle
    # Calculate the distance between the end-effector and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle

    # Milestone: Reward for being close enough to attempt grasping
    if dist_to_handle < 0.05:
        reward += 1.0  # Milestone reward for being close to the handle

    # Stage 2: Grasping the handle
    # Check if the gripper is close enough to the handle to consider it grasped
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Milestone: Reward for correct gripper orientation
        ee_pose = self.robot.ee_pose
        handle_pose = self.cabinet.handle.pose
        orientation_diff = np.linalg.norm(ee_pose.q - handle_pose.q)
        if orientation_diff < 0.1:  # Threshold for correct orientation
            reward += 1.0  # Reward for correct orientation

        # Milestone: Reward for gripper openness (encourage closing the gripper)
        gripper_openness = self.robot.gripper_openness
        if gripper_openness < 0.1:  # Threshold for closed gripper
            reward += 1.0  # Reward for closing the gripper

    # Stage 3: Opening the door
    if is_grasped:
        # Milestone: Reward for significant door opening progress
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        if door_progress > 0.5:  # Threshold for significant progress
            reward += 2.0  # Reward for significant door opening progress

    return reward