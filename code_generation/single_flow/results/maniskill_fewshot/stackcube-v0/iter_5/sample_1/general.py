import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for grasping Cube A
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Reward for placing Cube A on Cube B
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is placed on Cube B
            reward += 2.0  # Reward for successful placement

            # Step 3: Reward for precise alignment
            horizontal_dist = np.linalg.norm(self.cubeA.pose.p[:2] - self.cubeB.pose.p[:2])
            if horizontal_dist < 0.01:  # If Cube A is centered on Cube B
                reward += 0.5  # Reward for precise alignment

    # Step 4: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 5: Penalize unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    # Step 6: Reward for minimizing the time taken to complete the task
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 0.1 * (1 - self.current_step / self.max_steps)  # Reward for completing the task quickly

    return reward