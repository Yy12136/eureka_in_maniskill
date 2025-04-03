import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Get the current state of the drawer
    current_qpos = self.cabinet.handle.qpos
    target_qpos = self.cabinet.handle.target_qpos

    # Check if the task is completed
    is_task_completed = current_qpos >= target_qpos

    if is_task_completed:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Navigation to the cabinet
    # Calculate the distance between the mobile base and the cabinet
    base_to_cabinet_dist = np.linalg.norm(self.robot.base_position - self.cabinet.pose.p[:2])
    navigation_reward = 1 - np.tanh(5 * base_to_cabinet_dist)
    reward += navigation_reward

    # Stage 2: Reaching the drawer handle
    # Get the end-effector position and the handle's point cloud
    ee_pos = self.robot.ee_pose.p
    handle_pcd = self.cabinet.handle.get_world_pcd()

    # Calculate the minimum distance between the end-effector and the handle
    ee_to_handle_dist = cdist(ee_pos.reshape(1, 3), handle_pcd).min()
    reaching_reward = 1 - np.tanh(5 * ee_to_handle_dist)
    reward += reaching_reward

    # Stage 3: Grasping the handle
    # Check if the handle is grasped
    is_grasped = self.robot.check_grasp(self.cabinet.handle, max_angle=30)
    if is_grasped:
        reward += 1.0  # Reward for grasping the handle

    # Stage 4: Pulling the drawer
    if is_grasped:
        # Calculate the progress of pulling the drawer
        pull_progress = current_qpos / target_qpos
        pull_reward = np.tanh(5 * pull_progress)
        reward += pull_reward

    # Penalize unnecessary movements
    # Penalize high base velocity
    base_velocity_penalty = np.linalg.norm(self.robot.base_velocity)
    reward -= 0.1 * base_velocity_penalty

    # Penalize high joint velocities
    joint_velocity_penalty = np.linalg.norm(self.robot.qvel)
    reward -= 0.1 * joint_velocity_penalty

    return reward