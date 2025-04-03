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
    # Calculate the mean distance between the grippers and the chair
    dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).mean()
    # Reward for reducing the distance to the chair
    reward += -0.5 * dist_to_chair

    # Stage 2: Grasp the Chair
    # If the grippers are close enough to the chair, encourage grasping
    if dist_to_chair < 0.1:
        # Reward for maintaining proximity to the chair
        reward += 0.2 * (1 - dist_to_chair)

    # Stage 3: Chair Stability
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Calculate the angular and linear velocity of the chair
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    chair_linear_velocity = np.linalg.norm(self.root_link.velocity)
    # Penalize any tilt, angular velocity, and excessive linear velocity of the chair
    reward += -0.3 * chair_tilt - 0.2 * chair_angular_velocity - 0.1 * chair_linear_velocity

    # Stage 4: Push the Chair to the Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Calculate the direction of the chair's movement
    chair_velocity = self.root_link.velocity[:2]
    target_direction = self.target_xy - self.root_link.pose.p[:2]
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    # Reward for pushing the chair in the direction of the target
    push_efficiency = np.dot(chair_velocity, target_direction_normalized)
    # Penalize any lateral movement that deviates from the target direction
    lateral_movement = np.linalg.norm(chair_velocity - push_efficiency * target_direction_normalized)
    reward += -0.5 * dist_to_target + 0.2 * push_efficiency - 0.1 * lateral_movement

    # Stage 5: Stabilize the Chair at the Target
    # If the chair is close to the target, ensure it remains stable
    if dist_to_target < 0.1:
        reward += -0.3 * chair_tilt - 0.2 * chair_angular_velocity

    # Penalize unnecessary movement of the robot's base
    base_velocity = np.linalg.norm(self.robot.base_velocity)
    reward += -0.1 * base_velocity

    # Action Regularization
    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward += -0.1 * action_penalty

    return reward