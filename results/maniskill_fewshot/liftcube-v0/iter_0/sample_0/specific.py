import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action):
    reward = 0.0

    # Get the current position of cube A
    cubeA_pos = self.obj.pose.p
    # Get the initial height of cube A (assuming it's placed on a flat surface)
    initial_height = cubeA_pos[2]
    # Target height is 0.2 meters above the initial height
    target_height = initial_height + self.goal_height

    # Check if cube A is lifted to the target height
    is_lifted = cubeA_pos[2] >= target_height
    # Check if cube A is static (not moving)
    is_cubeA_static = check_actor_static(self.obj)
    # Check if the robot is holding cube A
    is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

    # Success condition: cube A is lifted, static, and grasped
    success = is_lifted and is_cubeA_static and is_grasped

    if success:
        reward += 10.0  # Large reward for completing the task
        return reward

    # Stage 1: Reaching cube A
    tcp_to_cubeA_dist = np.linalg.norm(self.tcp.pose.p - cubeA_pos)
    reaching_reward = 1 - np.tanh(5.0 * tcp_to_cubeA_dist)
    reward += reaching_reward

    # Stage 2: Grasping cube A
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # Stage 3: Lifting cube A
    if is_grasped:
        # Calculate the height difference between cube A and the target height
        height_diff = target_height - cubeA_pos[2]
        # Reward for lifting cube A closer to the target height
        lifting_reward = 1 - np.tanh(5.0 * height_diff)
        reward += lifting_reward

    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward