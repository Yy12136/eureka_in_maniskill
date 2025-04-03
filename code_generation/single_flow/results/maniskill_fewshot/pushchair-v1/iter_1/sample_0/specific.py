import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Get chair and target positions
    chair_pos = self.chair.pose.p[:2]  # XY position of the chair
    target_pos = self.target_xy  # XY target position
    chair_to_target_dist = np.linalg.norm(chair_pos - target_pos)  # Distance between chair and target

    # Check if the chair is at the target location
    is_chair_at_target = chair_to_target_dist <= 0.05  # Tolerance for target position
    is_chair_static = self.chair.check_static()  # Check if the chair is static
    success = is_chair_at_target and is_chair_static

    if success:
        reward += 15.0  # Large reward for completing the task
        return reward

    # Milestone 1: Approach the chair
    # Calculate the distance between the robot's base and the chair
    base_to_chair_dist = np.linalg.norm(self.robot.base_position - chair_pos)
    approaching_reward = 1 - np.tanh(5 * base_to_chair_dist)
    reward += approaching_reward * 0.5

    # Milestone 2: Stabilize the chair
    # Calculate the tilt of the chair (angle between chair's z-axis and world z-axis)
    z_axis_chair = self.chair.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])  # Angle in radians
    tilt_reward = 1 - np.tanh(5 * chair_tilt)
    reward += tilt_reward * 0.5

    # Milestone 3: Push the chair toward the target
    # Calculate the distance between the chair and the target
    pushing_reward = 1 - np.tanh(5 * chair_to_target_dist)
    reward += pushing_reward * 1.0

    # Milestone 4: Ensure the chair remains stable during pushing
    # Penalize if the chair is moving too fast (linear or angular velocity)
    chair_vel_penalty = np.linalg.norm(self.chair.velocity) + np.linalg.norm(self.chair.angular_velocity)
    reward -= chair_vel_penalty * 0.2

    # Milestone 5: Encourage the robot to use both arms effectively
    # Calculate the distance between each arm's gripper and the chair's surface
    ee_coords = self.robot.get_ee_coords()  # Get gripper finger positions
    chair_pcd = self.chair.get_pcd()  # Get chair's point cloud
    gripper_to_chair_dists = cdist(ee_coords.reshape(-1, 3), chair_pcd).min(axis=1).reshape(2, 4).mean(axis=1)
    gripper_reward = 1 - np.tanh(5 * gripper_to_chair_dists.mean())
    reward += gripper_reward * 0.5

    # Milestone 6: Penalize excessive robot movement
    # Penalize if the robot's base is moving too fast
    base_vel_penalty = np.linalg.norm(self.robot.base_velocity)
    reward -= base_vel_penalty * 0.1

    return reward