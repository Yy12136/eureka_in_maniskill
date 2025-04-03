import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successfully grasping Cube A
    if not self.robot.check_grasp(self.cubeA, max_angle=30):
        gripper_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
        if gripper_to_cubeA_dist < 0.05 and self.robot.check_grasp(self.cubeA, max_angle=30):
            reward += 1.0  # Reward for successful grasp

    # Step 2: Reward for lifting Cube A above Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        if self.cubeA.pose.p[2] > self.cubeB.pose.p[2] + 2 * self.cube_half_size:
            reward += 1.0  # Reward for lifting Cube A above Cube B

    # Step 3: Reward for placing Cube A on Cube B stably
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p[:2] - self.cubeB.pose.p[:2])
        if cubeA_to_cubeB_dist < 0.01:  # If Cube A is centered on Cube B
            reward += 2.0  # Reward for stable placement and alignment

    # Step 4: Penalize task failure (e.g., dropping Cube A or moving Cube B)
    if not self.cubeA.check_static() or np.linalg.norm(self.cubeB.velocity) > 0.01:
        reward += -1.0  # Penalize for instability or unintended movement

    return reward