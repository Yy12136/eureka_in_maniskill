import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Stability (Global Penalty)
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Penalize chair falling over (tilt beyond a threshold)
    if chair_tilt > np.pi / 6:  # 30 degrees threshold
        reward -= 1.0  # Large penalty for chair falling

    # Stage 2: Task Completion (Global Reward)
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for reaching the target location
    if dist_to_target < 0.1:  # Threshold for task completion
        reward += 1.0  # Large reward for task completion

    # Stage 3: Chair Orientation at Target (Optional)
    # If the chair is close to the target, ensure it is oriented correctly
    if dist_to_target < 0.1:
        target_orientation = self.target_xy - self.root_link.pose.p[:2]
        target_orientation_normalized = target_orientation / np.linalg.norm(target_orientation)
        chair_orientation = self.root_link.pose.to_transformation_matrix()[:3, 0]  # Assuming x-axis is forward
        orientation_alignment = np.dot(chair_orientation[:2], target_orientation_normalized)
        if orientation_alignment > 0.9:  # Threshold for good alignment
            reward += 0.5  # Additional reward for correct orientation

    return reward