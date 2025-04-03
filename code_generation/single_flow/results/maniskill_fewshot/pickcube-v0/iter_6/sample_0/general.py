import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get positions
    gripper_pos = self.robot.ee_pose.p
    cubeA_pos = self.cubeA.pose.p
    goal_pos = self.goal_position

    # Calculate distances
    gripper_to_cubeA_dist = np.linalg.norm(gripper_pos - cubeA_pos)
    cubeA_to_goal_dist = np.linalg.norm(cubeA_pos - goal_pos)

    # Check if cube A is grasped
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)

    # Check if cube A is at the goal position
    is_cubeA_at_goal = cubeA_to_goal_dist < 0.01  # Tolerance for success
    is_cubeA_static = self.cubeA.check_static()
    success = is_cubeA_at_goal and is_cubeA_static and not is_grasped

    # Success reward (primary focus)
    if success:
        reward += 1000.0  # Large reward for task completion
        return reward

    # Stage 1: Grasping Cube A
    if not is_grasped:
        # Reward for successful grasp
        if is_grasped:
            reward += 100.0  # Significant reward for grasping
    else:
        # Stage 2: Lifting Cube A
        lifting_height = cubeA_pos[2] - self.cube_half_size
        if lifting_height > 0.1:  # Reward for lifting above a threshold
            reward += 50.0

        # Stage 3: Moving Cube A to Goal
        if cubeA_to_goal_dist < 0.5:  # Reward for moving closer to the goal
            reward += 25.0

    # Penalize dropping the cube
    if not is_grasped and lifting_height > 0.1:
        reward -= 200.0  # Large penalty for dropping the cube

    # Regularization of the robot's action (small penalty to encourage smoothness)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward