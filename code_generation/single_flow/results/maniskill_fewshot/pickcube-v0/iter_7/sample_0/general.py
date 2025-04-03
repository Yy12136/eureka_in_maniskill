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

    # Success reward (large reward for task completion)
    if success:
        reward += 1000.0  # Sparse reward for completing the task
        return reward

    # Stage 1: Grasping Cube A
    if not is_grasped:
        # Reward for successful grasp
        if is_grasped:
            reward += 100.0  # Sparse reward for grasping the cube

    # Stage 2: Lifting Cube A
    if is_grasped:
        # Encourage lifting cube A to a certain height
        lifting_height = cubeA_pos[2] - self.cube_half_size
        if lifting_height > 0.1:  # Threshold for lifting
            reward += 150.0  # Sparse reward for lifting the cube

    # Stage 3: Moving Cube A to Goal
    if is_grasped:
        # Encourage moving cube A toward the goal
        if cubeA_to_goal_dist < 0.5:  # Threshold for proximity to the goal
            reward += 200.0  # Sparse reward for moving the cube close to the goal

    # Stage 4: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Encourage gripper openness for releasing
        gripper_openness = self.robot.gripper_openness
        if gripper_openness > 0.9:  # Threshold for releasing
            reward += 250.0  # Sparse reward for releasing the cube at the goal

    # Penalize dropping the cube
    if not is_grasped and cubeA_pos[2] > self.cube_half_size:
        reward -= 300.0  # Large penalty for dropping the cube

    return reward