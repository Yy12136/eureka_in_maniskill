import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached and Grasped
    # Check if the chair is within a small distance of the grippers
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.05:  # Chair is within grasping range
        reward += 1.0

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Chair is close to the target
        reward += 2.0

    # Stage 3: Chair Stable at Target
    # Check if the chair is stable (not tilted or moving) at the target
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 3.0  # Chair is stable at the target

    # Penalties for Failures
    # Penalize if the chair falls over (significant tilt)
    if chair_tilt > 0.5:  # Chair is significantly tilted
        reward -= 1.0

    # Penalize if the chair is not in contact with the ground
    if not self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2):
        reward -= 1.0

    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward -= 0.1 * action_penalty

    return reward