import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached
    # Check if the robot's end-effectors are close to the chair
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.1:  # Threshold for "reached" the chair
        reward += 1.0  # Sparse reward for reaching the chair

    # Stage 2: Chair Pushed to Target
    # Check if the chair is close to the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Threshold for "reached" the target
        reward += 2.0  # Sparse reward for pushing the chair to the target

    # Stage 3: Chair Stability
    # Check if the chair is stable (not tilted or moving)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    if chair_tilt < 0.1 and chair_angular_velocity < 0.1:  # Threshold for stability
        reward += 1.0  # Sparse reward for maintaining stability

    # Stage 4: Task Completion
    # Check if the chair is at the target and stable
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 3.0  # Sparse reward for completing the task

    return reward