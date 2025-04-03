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

    # Sparse reaching reward: only reward when the end-effector is close to the handle
    if dist_ee_to_handle < 0.05:
        reward += 1.0

    # Check if the handle is grasped
    is_grasped = self.robot.check_grasp(self.cabinet.handle)
    if is_grasped:
        reward += 2.0

        # Calculate the current qpos of the drawer
        current_qpos = self.cabinet.handle.qpos
        target_qpos = self.cabinet.handle.target_qpos

        # Sparse opening reward: only reward when the drawer is opened beyond the target qpos
        if current_qpos >= target_qpos:
            reward += 5.0

    # Success condition
    if current_qpos >= target_qpos:
        reward += 10.0

    return reward