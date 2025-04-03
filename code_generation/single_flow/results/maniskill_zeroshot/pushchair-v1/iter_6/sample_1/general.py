import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached and Grasped
    # Check if the chair is within a certain distance from the grippers
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.1:  # Chair is within reach
        reward += 1.0

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Chair is at the target
        reward += 2.0

    # Stage 3: Chair Stability
    # Check if the chair is stable (not tilted or moving)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    if chair_tilt < 0.1 and chair_angular_velocity < 0.1:  # Chair is stable
        reward += 1.0

    # Stage 4: Chair Orientation at Target
    # Check if the chair is oriented correctly at the target
    if dist_to_target < 0.1:
        target_orientation = self.target_xy - self.root_link.pose.p[:2]
        target_orientation_normalized = target_orientation / np.linalg.norm(target_orientation)
        chair_orientation = self.root_link.pose.to_transformation_matrix()[:3, 0]  # Assuming x-axis is forward
        orientation_alignment = np.dot(chair_orientation[:2], target_orientation_normalized)
        if orientation_alignment > 0.9:  # Chair is well-oriented
            reward += 1.0

    # Penalties
    # Penalize if the chair falls over or is not in contact with the ground
    if chair_tilt > 0.2 or not self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2):
        reward -= 2.0

    return reward