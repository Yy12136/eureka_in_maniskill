import numpy as np

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
                self.task_completed = True  # Mark task as completed

    # Step 4: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 5: Penalize unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    return reward