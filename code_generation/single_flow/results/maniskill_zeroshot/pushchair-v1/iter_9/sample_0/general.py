import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached
    # Check if the robot is close to the chair
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.1:  # Threshold for "close enough"
        reward += 0.5  # Reward for reaching the chair

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Threshold for "close enough"
        reward += 1.0  # Reward for pushing the chair to the target

    # Stage 3: Chair Stabilized at Target
    # Check if the chair is stable (no tilt or angular velocity)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0  # Reward for stabilizing the chair

    # Stage 4: Task Completion
    # Check if the chair is exactly at the target and stable
    if dist_to_target < 0.05 and chair_tilt < 0.05 and chair_angular_velocity < 0.05:
        reward += 2.0  # Large reward for task completion

    return reward