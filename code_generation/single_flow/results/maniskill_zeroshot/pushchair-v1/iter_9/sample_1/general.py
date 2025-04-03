import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the Chair
    # Get the point cloud of the chair
    chair_pcd = self.env.env._get_chair_pcd()
    # Get the end-effector coordinates of both arms
    ee_coords = self.agent.get_ee_coords()
    # Calculate the mean distance between the grippers and the chair
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    # Reward for reducing the distance to the chair
    if dist_to_chair < 0.1:
        reward += 0.1

    # Stage 2: Push the Chair to the Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for reducing the distance to the target
    if dist_to_target < 0.5:
        reward += 0.2
    if dist_to_target < 0.2:
        reward += 0.3

    # Stage 3: Task Completion
    # Ensure the chair is exactly at the target position and stable
    if dist_to_target < 0.05:
        # Check if the chair is stable (low tilt and angular velocity)
        z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])
        chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
        if chair_tilt < 0.1 and chair_angular_velocity < 0.1:
            reward += 1.0

    return reward