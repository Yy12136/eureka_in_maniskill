import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the Chair
    # Get the point cloud of the chair
    chair_pcd = self.env.env._get_chair_pcd()
    # Get the end-effector coordinates of both arms
    ee_coords = self.agent.get_ee_coords()
    # Calculate the minimum distance between the grippers and the chair
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min()
    # Reward for reducing the distance to the chair
    reward += -0.5 * dist_to_chair

    # Stage 2: Chair Stability
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Penalize any tilt of the chair
    reward += -0.5 * chair_tilt

    # Stage 3: Push the Chair to the Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for reducing the distance to the target
    reward += -0.5 * dist_to_target

    # Stage 4: Stabilize the Chair at the Target
    # If the chair is close to the target, ensure it remains stable
    if dist_to_target < 0.1:
        reward += -0.5 * chair_tilt

    # Action Regularization
    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward += -0.1 * action_penalty

    return reward