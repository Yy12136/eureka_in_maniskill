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

    # Stage 2: Chair Stability
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Calculate the angular velocity of the chair
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    # Penalize any tilt and angular velocity of the chair
    reward += -0.3 * chair_tilt - 0.2 * chair_angular_velocity

    # Stage 3: Push the Chair to the Target
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Calculate the direction of the chair's movement
    chair_velocity = self.root_link.velocity[:2]
    target_direction = self.target_xy - self.root_link.pose.p[:2]
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    # Reward for pushing the chair in the direction of the target
    push_efficiency = np.dot(chair_velocity, target_direction_normalized)
    reward += -0.5 * dist_to_target + 0.2 * push_efficiency

    # Stage 4: Stabilize the Chair at the Target
    # If the chair is close to the target, ensure it remains stable
    if dist_to_target < 0.1:
        reward += -0.3 * chair_tilt - 0.2 * chair_angular_velocity

    # Stage 5: Smooth Movement of the Robot
    # Penalize large actions to encourage smooth movements
    action_penalty = np.sum(np.square(action))
    reward += -0.1 * action_penalty

    # Stage 6: Base Movement Efficiency
    # Calculate the direction of the base's movement
    base_velocity = self.robot.base_velocity
    base_direction = self.root_link.pose.p[:2] - self.robot.base_position
    base_direction_normalized = base_direction / np.linalg.norm(base_direction)
    # Reward for moving the base in the direction of the chair
    base_efficiency = np.dot(base_velocity, base_direction_normalized)
    reward += 0.1 * base_efficiency

    # Stage 7: Chair Orientation at Target
    # If the chair is close to the target, ensure it is oriented correctly
    if dist_to_target < 0.1:
        target_orientation = self.target_xy - self.root_link.pose.p[:2]
        target_orientation_normalized = target_orientation / np.linalg.norm(target_orientation)
        chair_orientation = self.root_link.pose.to_transformation_matrix()[:3, 0]  # Assuming x-axis is forward
        orientation_alignment = np.dot(chair_orientation[:2], target_orientation_normalized)
        reward += 0.1 * orientation_alignment

    # Stage 8: Chair Grasp Stability
    # Ensure that the chair is grasped stably by both arms
    grasp_stability = self.robot.get_grasp_stability()
    reward += 0.1 * grasp_stability

    # Stage 9: Chair Contact with Ground
    # Penalize if the chair is not in contact with the ground
    if not self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2):
        reward += -0.2

    # Stage 10: Chair Position Precision
    # If the chair is very close to the target, ensure it is precisely positioned
    if dist_to_target < 0.05:
        reward += -0.5 * dist_to_target

    return reward