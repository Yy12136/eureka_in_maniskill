import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached
    # Check if the robot has successfully approached the chair
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.1:  # Threshold for successful approach
        reward += 1.0  # Sparse reward for reaching the chair

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Threshold for task completion
        reward += 2.0  # Sparse reward for pushing the chair to the target

    # Stage 3: Chair Stability
    # Penalize if the chair falls over or becomes unstable
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    if chair_tilt > np.pi / 6:  # Threshold for significant tilt (30 degrees)
        reward += -1.0  # Sparse penalty for instability

    return reward