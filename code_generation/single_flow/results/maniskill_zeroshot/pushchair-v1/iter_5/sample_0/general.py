import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reached and Grasped
    # Check if the chair is within a threshold distance of the end-effectors
    chair_pcd = self.env.env._get_chair_pcd()
    ee_coords = self.agent.get_ee_coords()
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    if dist_to_chair < 0.05:  # Threshold for successful grasp
        reward += 0.2

    # Stage 2: Chair Stability During Movement
    # Check if the chair remains stable (minimal tilt and angular velocity)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    if chair_tilt < 0.1 and chair_angular_velocity < 0.1:  # Threshold for stability
        reward += 0.2

    # Stage 3: Chair Reached Target Location
    # Check if the chair is within a threshold distance of the target
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    if dist_to_target < 0.1:  # Threshold for task completion
        reward += 0.4

    # Stage 4: Chair Stability at Target
    # Ensure the chair is stable at the target location
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 0.2

    # Stage 5: Task Completion Bonus
    # Add a large bonus for successfully completing the task
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0

    return reward