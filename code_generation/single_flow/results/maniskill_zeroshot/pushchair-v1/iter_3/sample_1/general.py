import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Task Completion (Primary Reward)
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # If the chair is within a small radius of the target, give a large reward
    if dist_to_target < 0.1:
        reward += 10.0  # Large reward for task completion

    # Stage 2: Chair Stability (Secondary Reward)
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # If the chair is upright, give a small reward
    if chair_tilt < np.deg2rad(10):  # Within 10 degrees of upright
        reward += 1.0

    # Stage 3: Penalize Chair Falling (Negative Reward)
    # If the chair falls over, give a large penalty
    if chair_tilt > np.deg2rad(45):  # Chair is considered fallen
        reward -= 10.0

    return reward