import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Approach Cube A
    # Calculate the distance between the gripper and cube A
    gripper_pos = self.robot.ee_pose.p
    cubeA_pos = self.cubeA.pose.p
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)

    # Reward for reducing the distance to cube A
    reward += 0.5 * (1 - np.tanh(10 * distance_to_cubeA))

    # Stage 2: Grasp Cube A
    if distance_to_cubeA < 0.05:  # If the gripper is close enough to cube A
        # Check if the gripper is grasping cube A
        if self.robot.check_grasp(self.cubeA, max_angle=30):
            # Reward for successful grasp
            reward += 0.3

            # Stage 3: Lift Cube A
            # Calculate the height difference between cube A and the goal height
            cubeA_height = self.cubeA.pose.p[2]
            height_difference = abs(cubeA_height - self.goal_height)

            # Reward for lifting cube A to the desired height
            reward += 0.2 * (1 - np.tanh(10 * height_difference))

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = 0.1 * np.linalg.norm(action)
    reward -= action_penalty

    return reward