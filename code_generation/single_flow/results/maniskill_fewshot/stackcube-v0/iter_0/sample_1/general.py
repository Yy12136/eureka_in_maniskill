import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for grasping Cube A
    if not self.cubeA.is_grasped and self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for successful grasp
        self.cubeA.is_grasped = True  # Mark Cube A as grasped

    # Step 2: Reward for placing Cube A on Cube B
    if self.cubeA.is_grasped:
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is close to Cube B
            reward += 1.0  # Reward for placing Cube A near Cube B

            # Step 3: Reward for stable placement and release
            if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
                reward += 2.0  # Reward for stable placement and release
                self.cubeA.is_placed = True  # Mark Cube A as placed

    # Step 4: Reward for precise alignment of Cube A on Cube B
    if self.cubeA.is_placed:
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
        if horizontal_dist < 0.01:  # If Cube A is centered on Cube B
            reward += 0.5  # Reward for precise alignment

    # Step 5: Reward for keeping Cube A upright
    if self.cubeA.is_placed:
        cubeA_rotation = self.cubeA.pose.to_transformation_matrix()[:3, :3]
        desired_rotation = np.eye(3)  # Desired rotation is upright
        rotation_diff = np.linalg.norm(cubeA_rotation - desired_rotation)
        if rotation_diff < 0.1:  # If Cube A is upright
            reward += 0.5  # Reward for keeping Cube A upright

    # Step 6: Reward for completing the task quickly
    if self.cubeA.is_placed:
        reward += 0.1 * (1 - self.current_step / self.max_steps)  # Reward for completing the task quickly

    return reward