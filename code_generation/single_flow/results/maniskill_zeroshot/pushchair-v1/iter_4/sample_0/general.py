import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached
    # Get the point cloud of the chair
    chair_pcd = self.env.env._get_chair_pcd()
    # Get the end-effector coordinates of both arms
    ee_coords = self.agent.get_ee_coords()
    # Calculate the mean distance between the grippers and the chair
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    # Reward for reaching the chair
    if dist_to_chair < 0.1:
        reward += 1.0

    # Stage 2: Chair Pushed to Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for pushing the chair to the target
    if dist_to_target < 0.1:
        reward += 2.0

    # Stage 3: Chair Stable at Target
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Calculate the angular velocity of the chair
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    # Reward for keeping the chair stable at the target
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0

    # Stage 4: Chair Orientation at Target
    # If the chair is close to the target, ensure it is oriented correctly
    if dist_to_target < 0.1:
        target_orientation = self.target_xy - self.root_link.pose.p[:2]
        target_orientation_normalized = target_orientation / np.linalg.norm(target_orientation)
        chair_orientation = self.root_link.pose.to_transformation_matrix()[:3, 0]  # Assuming x-axis is forward
        orientation_alignment = np.dot(chair_orientation[:2], target_orientation_normalized)
        # Reward for correct orientation
        if orientation_alignment > 0.9:
            reward += 1.0

    return reward