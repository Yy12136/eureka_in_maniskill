import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Get relevant poses and positions
    ee_pose = self.robot.ee_pose
    handle_pose = self.cabinet.handle.pose
    handle_pcd = self.cabinet.handle.get_world_pcd()
    ee_coords = self.robot.get_ee_coords()

    # Calculate distance between end-effector and handle
    dist_ee_to_handle = cdist(ee_coords, handle_pcd).min()
    reaching_reward = 1 - np.tanh(5 * dist_ee_to_handle)
    reward += reaching_reward

    # Check if the handle is grasped
    is_grasped = self.robot.check_grasp(self.cabinet.handle)
    if is_grasped:
        reward += 1.0

        # Calculate the current qpos of the drawer
        current_qpos = self.cabinet.handle.qpos
        target_qpos = self.cabinet.handle.target_qpos

        # Reward for opening the drawer
        opening_reward = np.tanh(10 * (current_qpos - target_qpos))
        reward += opening_reward

        # Success condition
        if current_qpos >= target_qpos:
            reward += 5.0

    # Regularization of the robot's action
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    # Penalize base velocity to encourage stable movement
    base_velocity_penalty = -0.01 * np.linalg.norm(self.robot.base_velocity)
    reward += base_velocity_penalty

    # Penalize joint velocity to encourage smooth movement
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.robot.qvel)
    reward += joint_velocity_penalty

    return reward