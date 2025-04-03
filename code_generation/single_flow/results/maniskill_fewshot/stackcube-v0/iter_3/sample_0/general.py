import numpy as np

def compute_sparse_reward(self) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Check if Cube A is on Cube B and stable
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])

        # Step 1a: Reward for placing Cube A on Cube B
        if horizontal_dist < 0.01:  # Cube A is centered on Cube B
            reward += 10.0  # Large reward for successful task completion
        else:
            reward += 5.0  # Smaller reward for partial success (Cube A is on Cube B but not centered)

    # Step 2: Penalize any movement of Cube B during the task
    if self.cubeB.check_static():
        reward += -0.1 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    # Step 3: Penalize excessive time taken to complete the task
    reward += -0.01 * self.current_step  # Penalize time steps linearly

    return reward