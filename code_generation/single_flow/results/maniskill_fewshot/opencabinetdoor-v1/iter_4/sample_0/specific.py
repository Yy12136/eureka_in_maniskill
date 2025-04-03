import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Approaching the cabinet
    # Calculate the distance between the end-effector and the cabinet's point cloud
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    cabinet_pcd = self.cabinet.get_pcd()  # Get the point cloud of the cabinet in the world frame
    dist_to_cabinet = cdist(ee_coords, cabinet_pcd).min(axis=1).mean()  # Minimum distance between gripper and cabinet
    approaching_reward = 1 - np.tanh(5.0 * dist_to_cabinet)  # Reward for getting closer to the cabinet
    reward += approaching_reward

    # Milestone: Reward for being close enough to the cabinet
    if dist_to_cabinet < 0.1:
        reward += 1.0  # Milestone reward for being close to the cabinet

    # Stage 2: Approaching the cabinet handle
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle
    reaching_reward = 1 - np.tanh(5.0 * dist_to_handle)  # Reward for getting closer to the handle
    reward += reaching_reward

    # Milestone: Reward for being close enough to attempt grasping
    if dist_to_handle < 0.05:
        reward += 1.0  # Milestone reward for being close to the handle

    # Stage 3: Grasping the handle
    # Check if the gripper is close enough to the handle to consider it grasped
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Milestone: Reward for correct gripper orientation
        ee_pose = self.robot.ee_pose
        handle_pose = self.cabinet.handle.pose
        orientation_diff = np.linalg.norm(ee_pose.q - handle_pose.q)
        orientation_reward = 1 - np.tanh(5.0 * orientation_diff)
        reward += orientation_reward

        # Milestone: Reward for gripper openness (encourage closing the gripper)
        gripper_openness = self.robot.gripper_openness
        gripper_reward = 1 - gripper_openness  # Reward for closing the gripper
        reward += gripper_reward

        # Stage 4: Opening the door
        # Reward based on how much the door has been opened
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        opening_reward = np.clip(door_progress, 0.0, 1.0)  # Reward proportional to door opening progress
        reward += 3.0 * opening_reward

        # Penalize large actions to encourage smooth movements
        action_penalty = -0.01 * np.linalg.norm(action)
        reward += action_penalty

        # Penalize unnecessary mobile base movement
        base_velocity_penalty = -0.01 * np.linalg.norm(self.robot.base_velocity)
        reward += base_velocity_penalty

    # Milestone: Encourage the robot to maintain a stable base position
    base_position_diff = np.linalg.norm(self.robot.base_position - self.robot.initial_base_position)
    base_stability_reward = 1 - np.tanh(5.0 * base_position_diff)
    reward += base_stability_reward

    # Milestone: Reward for maintaining a stable arm configuration
    joint_position_diff = np.linalg.norm(self.robot.qpos - self.robot.initial_qpos)
    arm_stability_reward = 1 - np.tanh(5.0 * joint_position_diff)
    reward += arm_stability_reward

    return reward