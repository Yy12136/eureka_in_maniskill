import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successfully grasping Cube A
    if not self.has_grasped and self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for successful grasp
        self.has_grasped = True  # Mark that Cube A has been grasped

    # Step 2: Reward for lifting Cube A above Cube B
    if self.has_grasped and self.cubeA.pose.p[2] > self.cubeB.pose.p[2] + self.cube_half_size:
        reward += 1.0  # Reward for lifting Cube A

    # Step 3: Reward for placing Cube A on Cube B
    if self.has_grasped and np.linalg.norm(self.cubeA.pose.p[:2] - self.cubeB.pose.p[:2]) < 0.01:
        reward += 1.0  # Reward for placing Cube A near Cube B

    # Step 4: Reward for stable placement and releasing Cube A
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 2.0  # Reward for stable placement and release
        self.task_completed = True  # Mark the task as completed

    # Step 5: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 6: Penalize unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)

    return reward