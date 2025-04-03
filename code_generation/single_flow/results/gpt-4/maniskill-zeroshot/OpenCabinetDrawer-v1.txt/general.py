import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Encourage the robot to approach the cabinet
    # Calculate the distance between the robot's base and the cabinet
    base_to_cabinet_distance = np.linalg.norm(self.robot.base_position - self.cabinet.pose.p[:2])
    # Reward for reducing the distance to the cabinet
    reward += -0.1 * base_to_cabinet_distance

    # Stage 2: Encourage the robot to align the gripper with the handle
    # Get the gripper's end-effector pose in the world frame
    ee_pose = self.robot.ee_pose
    # Get the handle's pose in the world frame
    handle_pose = self.cabinet.handle.pose
    # Calculate the distance between the gripper and the handle
    gripper_to_handle_distance = np.linalg.norm(ee_pose.p - handle_pose.p)
    # Reward for reducing the distance to the handle
    reward += -0.2 * gripper_to_handle_distance

    # Stage 3: Encourage the robot to grasp the handle
    # Calculate the distance between the gripper fingers and the handle's point cloud
    ee_coords = self.robot.get_ee_coords()
    handle_pcd = self.cabinet.handle.get_world_pcd()
    gripper_to_handle_pcd_distance = cdist(ee_coords, handle_pcd).min(axis=1).mean()
    # Reward for reducing the distance to the handle's point cloud
    reward += -0.3 * gripper_to_handle_pcd_distance

    # Stage 4: Encourage the robot to pull the drawer open
    # Calculate the difference between the current qpos and the target qpos
    qpos_diff = self.cabinet.handle.target_qpos - self.cabinet.handle.qpos
    # Reward for increasing the qpos (pulling the drawer open)
    reward += 0.4 * qpos_diff

    # Stage 5: Task completion bonus
    if self.cabinet.handle.qpos >= self.cabinet.handle.target_qpos:
        reward += 1.0  # Large bonus for completing the task

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward