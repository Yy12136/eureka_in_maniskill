import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successful grasp of Cube A
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Reward for placing Cube A near Cube B
    cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
    if cubeA_to_cubeB_dist < 0.05:
        reward += 1.0  # Reward for placing Cube A near Cube B

    # Step 3: Reward for stable placement and release of Cube A on Cube B
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 2.0  # Reward for stable placement and release

    # Step 4: Reward for precise alignment of Cube A on Cube B
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
        if horizontal_dist < 0.01:
            reward += 0.5  # Reward for precise alignment

    # Step 5: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 6: Penalize excessive joint velocities for smoother movements
    reward += -0.01 * np.linalg.norm(self.robot.qvel)

    # Step 7: Penalize any unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    return reward