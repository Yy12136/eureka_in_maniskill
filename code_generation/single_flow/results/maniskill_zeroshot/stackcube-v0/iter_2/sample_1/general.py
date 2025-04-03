import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Check if Cube A is on Cube B and static
    cubeA_pos = self.cubeA.pose.p
    cubeB_pos = self.cubeB.pose.p
    horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
    height_diff = cubeA_pos[2] - cubeB_pos[2]

    # Check if Cube A is on top of Cube B and not grasped
    if horizontal_dist < 0.02 and abs(height_diff - 0.02) < 0.01 and check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
        # Large reward for task completion
        reward += 10.0
        return reward

    # Check if Cube A is grasped and lifted above Cube B
    if self.agent.check_grasp(self.cubeA, max_angle=30) and height_diff > 0.02 * 2:
        # Small reward for lifting Cube A
        reward += 1.0

    # Check if Cube A is grasped
    if self.agent.check_grasp(self.cubeA, max_angle=30):
        # Small reward for grasping Cube A
        reward += 0.5

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward