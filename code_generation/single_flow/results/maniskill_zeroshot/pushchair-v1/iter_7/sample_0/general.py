import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Reaches Target Location
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for the chair being close to the target (e.g., within 0.1 meters)
    if dist_to_target < 0.1:
        reward += 1.0  # Large reward for reaching the target

    # Stage 2: Chair Stability at Target
    # Check if the chair is stable (not tilted or moving significantly)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    chair_angular_velocity = np.linalg.norm(self.root_link.angular_velocity)
    # If the chair is stable at the target, reward it
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 1.0  # Additional reward for stability

    # Stage 3: Task Completion
    # If the chair is at the target and stable, the task is complete
    if dist_to_target < 0.1 and chair_tilt < 0.1 and chair_angular_velocity < 0.1:
        reward += 2.0  # Large reward for task completion

    # Stage 4: Penalize Chair Falling Over
    # If the chair falls over (tilt exceeds a threshold), penalize heavily
    if chair_tilt > 0.5:
        reward -= 2.0  # Heavy penalty for failure

    return reward