import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the Chair
    # Get the point cloud of the chair
    chair_pcd = self.env.env._get_chair_pcd()
    # Get the end-effector coordinates of both arms
    ee_coords = self.agent.get_ee_coords()
    # Calculate the minimum distance between the grippers and the chair
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min()
    # Reward for getting close to the chair
    if dist_to_chair < 0.1:  # Threshold for proximity
        reward += 1.0  # Sparse reward for approaching the chair

    # Stage 2: Push the Chair to the Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for moving the chair closer to the target
    if dist_to_target < 0.5:  # Threshold for pushing progress
        reward += 1.0  # Sparse reward for pushing the chair toward the target

    # Stage 3: Stabilize the Chair at the Target
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Check if the chair is stable and at the target
    if dist_to_target < 0.1 and chair_tilt < 0.1:  # Threshold for stability and proximity
        reward += 2.0  # Sparse reward for completing the task

    # Action Regularization (optional, for smoothness)
    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward += -0.01 * action_penalty  # Small penalty to avoid excessive actions

    return reward