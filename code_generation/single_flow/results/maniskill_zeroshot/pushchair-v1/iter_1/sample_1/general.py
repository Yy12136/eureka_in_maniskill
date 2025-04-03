import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached
    # Check if the grippers are close to the chair
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min()
    if dist_to_chair < 0.05:  # Threshold for "reached"
        reward += 1.0  # Reward for reaching the chair

    # Stage 2: Chair Pushed to Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Threshold for "reached target"
        reward += 2.0  # Reward for pushing the chair to the target

    # Stage 3: Chair Stability
    # Check if the chair is upright (tilt angle close to 0)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    if chair_tilt < 0.1:  # Threshold for "stable"
        reward += 1.0  # Reward for keeping the chair stable

    # Penalize Chair Falling
    # If the chair falls over, penalize heavily
    if chair_tilt > 0.5:  # Threshold for "fallen"
        reward -= 5.0  # Penalty for chair falling

    # Task Completion Bonus
    # If the chair is at the target and stable, give a large bonus
    if dist_to_target < 0.1 and chair_tilt < 0.1:
        reward += 5.0  # Bonus for completing the task

    return reward