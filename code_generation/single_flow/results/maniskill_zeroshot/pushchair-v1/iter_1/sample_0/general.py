import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get the chair's position and orientation
    chair_pos = self.root_link.pose.p[:2]  # 2D position (x, y)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]  # Chair's z-axis
    chair_tilt = np.arccos(z_axis_chair[2])  # Tilt angle from vertical

    # Get the target position
    target_pos = self.target_xy

    # Stage 1: Chair is within a small distance of the target
    dist_to_target = np.linalg.norm(chair_pos - target_pos)
    if dist_to_target < 0.1:  # Chair is close to the target
        reward += 1.0  # Reward for reaching the target
        if chair_tilt < 0.1:  # Chair is stable (minimal tilt)
            reward += 1.0  # Additional reward for stability
        return reward  # Task is complete, return the reward

    # Stage 2: Chair is upright and being pushed toward the target
    if chair_tilt < 0.2:  # Chair is mostly upright
        # Check if the chair is moving toward the target
        chair_velocity = self.root_link.velocity[:2]  # 2D velocity (x, y)
        direction_to_target = (target_pos - chair_pos) / (dist_to_target + 1e-6)
        movement_toward_target = np.dot(chair_velocity, direction_to_target)
        if movement_toward_target > 0:  # Chair is moving toward the target
            reward += 0.1  # Small reward for progress

    # Stage 3: Penalize chair falling over
    if chair_tilt > 0.5:  # Chair is significantly tilted
        reward -= 1.0  # Penalize instability

    return reward