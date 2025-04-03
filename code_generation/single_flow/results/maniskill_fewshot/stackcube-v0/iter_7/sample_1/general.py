import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for grasping Cube A
    if not self.cubeA.is_grasped and self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for successful grasp
        self.cubeA.is_grasped = True  # Mark Cube A as grasped

    # Step 2: Reward for lifting Cube A above Cube B
    if self.cubeA.is_grasped and self.cubeA.pose.p[2] > self.cubeB.pose.p[2] + self.cube_half_size:
        reward += 1.0  # Reward for lifting Cube A

    # Step 3: Reward for placing Cube A near Cube B
    if self.cubeA.is_grasped:
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p[:2] - self.cubeB.pose.p[:2])  # Horizontal distance
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is close to Cube B horizontally
            reward += 1.0  # Reward for placing Cube A near Cube B

    # Step 4: Reward for stable placement and release
    if self.cubeA.is_grasped and self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 2.0  # Reward for stable placement and release
        self.task_completed = True  # Mark task as completed

    # Step 5: Penalize unnecessary actions
    if not self.task_completed:
        reward += -0.01 * np.linalg.norm(action)  # Penalize large actions

    return reward