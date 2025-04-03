import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached and Stable
    # Check if the grippers are close to the chair
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    # Check if the chair is stable (tilt and angular velocity are small)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    # Reward for reaching and stabilizing the chair
    if dist_to_chair < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for reaching the target
    if dist_to_target < 0.1:
        reward += 1.0

    # Stage 3: Chair Stabilized at Target
    # Check if the chair is stable at the target location
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0

    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward += -0.1 * action_penalty

    return reward