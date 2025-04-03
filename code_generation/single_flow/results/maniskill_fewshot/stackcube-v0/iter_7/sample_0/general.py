import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successfully grasping Cube A
    if not self.cubeA.is_grasped and self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Significant reward for grasping
        self.cubeA.is_grasped = True  # Mark Cube A as grasped

    # Step 2: Reward for lifting Cube A above Cube B
    if self.cubeA.is_grasped and self.cubeA.pose.p[2] > self.cubeB.pose.p[2] + self.cube_half_size:
        reward += 1.0  # Significant reward for lifting

    # Step 3: Reward for placing Cube A on Cube B and releasing it
    if self.cubeA.is_grasped and np.linalg.norm(self.cubeA.pose.p[:2] - self.cubeB.pose.p[:2]) < 0.01:
        if self.cubeA.pose.p[2] <= self.cubeB.pose.p[2] + self.cube_half_size + 0.01:
            if not self.robot.check_grasp(self.cubeA, max_angle=30) and self.cubeA.check_static():
                reward += 2.0  # Significant reward for successful placement and release

    # Step 4: Penalize unnecessary movements of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Small penalty for Cube B movement

    # Step 5: Penalize excessive action magnitude for smoother movements
    reward += -0.01 * np.linalg.norm(action)

    return reward