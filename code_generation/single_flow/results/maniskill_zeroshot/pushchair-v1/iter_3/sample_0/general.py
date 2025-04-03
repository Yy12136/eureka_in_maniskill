import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Chair Stability
    # Calculate the tilt of the chair (angle between z-axis and vertical)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # Penalize significant tilt of the chair (e.g., > 15 degrees)
    if chair_tilt > np.radians(15):
        reward -= 1.0

    # Stage 2: Chair at Target Location
    # Calculate the distance between the chair and the target location
    dist_to_target = np.linalg.norm(self.root_link.pose.p[:2] - self.target_xy)
    # Reward for reaching the target location (e.g., within 10 cm)
    if dist_to_target < 0.1:
        reward += 1.0

    # Stage 3: Chair Stability at Target
    # If the chair is at the target, ensure it remains stable
    if dist_to_target < 0.1 and chair_tilt < np.radians(15):
        reward += 0.5

    # Stage 4: Task Completion
    # Check if the task is fully completed (chair at target and stable)
    if dist_to_target < 0.1 and chair_tilt < np.radians(15):
        reward += 1.0

    return reward