import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Check if the task is completed
    is_door_open = self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos
    if is_door_open:
        reward += 30.0  # Large reward for task completion
        return reward

    # Stage 1: Approaching the cabinet handle
    # Calculate the distance between the end-effector and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()  # Get the 3D positions of the gripper fingers
    handle_pcd = self.cabinet.handle.get_world_pcd()  # Get the point cloud of the handle in the world frame
    dist_to_handle = cdist(ee_coords, handle_pcd).min(axis=1).mean()  # Minimum distance between gripper and handle
    reaching_reward = 1 - np.tanh(5.0 * dist_to_handle)  # Reward for getting closer to the handle
    reward += reaching_reward

    # Milestone: Reward for being close enough to attempt grasping
    if dist_to_handle < 0.05:
        reward += 3.0  # Milestone reward for being close to the handle

    # Stage 2: Grasping the handle
    # Check if the gripper is close enough to the handle to consider it grasped
    is_grasped = dist_to_handle < 0.02  # Threshold for considering the handle grasped
    if is_grasped:
        reward += 5.0  # Reward for successful grasp

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

        # Milestone: Reward for aligning the gripper with the handle's grasp pose
        grasp_pose_diff = min([np.linalg.norm(ee_pose.p - grasp_pose.p) for grasp_pose in self.cabinet.handle.target_grasp_poses])
        grasp_alignment_reward = 1 - np.tanh(5.0 * grasp_pose_diff)
        reward += grasp_alignment_reward

        # Stage 3: Opening the door
        # Reward based on how much the door has been opened
        door_progress = self.cabinet.handle.qpos / self.cabinet.handle.target_qpos
        opening_reward = np.clip(door_progress, 0.0, 1.0)  # Reward proportional to door opening progress
        reward += 6.0 * opening_reward

        # Milestone: Reward for maintaining a smooth door opening velocity
        door_velocity = self.cabinet.handle.qvel
        velocity_reward = 1 - np.tanh(5.0 * np.abs(door_velocity - 0.1))  # Encourage a smooth velocity of 0.1
        reward += velocity_reward

        # Penalize large actions to encourage smooth movements
        action_penalty = -0.02 * np.linalg.norm(action)
        reward += action_penalty

        # Penalize unnecessary mobile base movement
        base_velocity_penalty = -0.02 * np.linalg.norm(self.robot.base_velocity)
        reward += base_velocity_penalty

    # Milestone: Encourage the robot to maintain a stable base position
    base_position_diff = np.linalg.norm(self.robot.base_position - self.robot.initial_base_position)
    base_stability_reward = 1 - np.tanh(5.0 * base_position_diff)
    reward += base_stability_reward

    # Milestone: Reward for maintaining a stable arm configuration
    joint_position_diff = np.linalg.norm(self.robot.qpos - self.robot.initial_qpos)
    arm_stability_reward = 1 - np.tanh(5.0 * joint_position_diff)
    reward += arm_stability_reward

    # Milestone: Reward for maintaining the mobile base's orientation
    base_orientation_diff = np.linalg.norm(self.robot.base_orientation - self.robot.initial_base_orientation)
    base_orientation_reward = 1 - np.tanh(5.0 * base_orientation_diff)
    reward += base_orientation_reward

    # Milestone: Reward for maintaining a stable gripper position relative to the handle
    if is_grasped:
        gripper_position_diff = np.linalg.norm(ee_pose.p - self.cabinet.handle.pose.p)
        gripper_stability_reward = 1 - np.tanh(5.0 * gripper_position_diff)
        reward += gripper_stability_reward

    # Milestone: Reward for maintaining a stable arm velocity
    arm_velocity_diff = np.linalg.norm(self.robot.qvel)
    arm_velocity_reward = 1 - np.tanh(5.0 * arm_velocity_diff)
    reward += arm_velocity_reward

    # Milestone: Reward for maintaining a stable base velocity
    base_velocity_diff = np.linalg.norm(self.robot.base_velocity)
    base_velocity_reward = 1 - np.tanh(5.0 * base_velocity_diff)
    reward += base_velocity_reward

    return reward