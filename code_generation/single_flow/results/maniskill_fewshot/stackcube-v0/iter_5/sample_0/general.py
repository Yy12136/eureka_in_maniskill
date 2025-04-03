import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successful grasp of Cube A
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 1.0  # Reward for grasping Cube A

    # Step 2: Reward for placing Cube A on Cube B stably
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
        vertical_dist = cubeA_center[2] - cubeB_center[2]

        # Check if Cube A is centered and stable on Cube B
        if horizontal_dist < 0.01 and abs(vertical_dist - self.cube_half_size) < 0.01:
            reward += 2.0  # Reward for stable and precise placement

    # Step 3: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 4: Penalize unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)

    # Step 5: Reward for minimizing the time taken to complete the task
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 0.1 * (1 - self.current_step / self.max_steps)  # Reward for completing the task quickly

    return reward