import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Check if Cube A is on top of Cube B and stable
    cubeA_pos = self.cubeA.pose.p
    cubeB_pos = self.cubeB.pose.p

    # Check if Cube A is on top of Cube B
    horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
    height_diff = cubeA_pos[2] - cubeB_pos[2]

    if horizontal_dist < 0.02 and height_diff > 0.02 and height_diff < 0.05:
        # Check if Cube A is static and not grasped
        if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
            # Reward for successfully placing and stabilizing Cube A
            reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward