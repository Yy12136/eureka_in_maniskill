import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Cube A is on top of Cube B
    cubeA_pos = self.cubeA.pose.p
    cubeB_pos = self.cubeB.pose.p

    # Check if Cube A is aligned with Cube B horizontally
    horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
    is_aligned = horizontal_dist < 0.02  # Threshold for alignment

    # Check if Cube A is above Cube B vertically
    height_diff = cubeA_pos[2] - cubeB_pos[2]
    is_above = height_diff > 0.02  # Threshold for being above

    # Milestone 2: Cube A is static and not grasped
    is_static = check_actor_static(self.cubeA)
    is_not_grasped = not self.agent.check_grasp(self.cubeA, max_angle=30)

    # Final reward: Only give a reward if all conditions are met
    if is_aligned and is_above and is_static and is_not_grasped:
        reward += 1.0  # Sparse reward for task completion

    return reward