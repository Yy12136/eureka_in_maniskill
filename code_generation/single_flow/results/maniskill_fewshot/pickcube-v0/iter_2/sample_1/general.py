import numpy as np

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

    # Success reward (sparse reward for task completion)
    if success:
        reward += 100.0  # Large reward for completing the task
        return reward

    # Stage 1: Reaching Cube A
    if not is_grasped:
        # Small reward for getting close to cube A
        if gripper_to_cubeA_dist < 0.1:
            reward += 10.0

    # Stage 2: Grasping Cube A
    if not is_grasped:
        # Reward for successful grasp
        if is_grasped:
            reward += 20.0

    # Stage 3: Lifting Cube A
    if is_grasped:
        # Reward for lifting cube A to a certain height
        lifting_height = cubeA_pos[2] - self.cube_half_size
        if lifting_height > 0.1:
            reward += 10.0

    # Stage 4: Moving Cube A to Goal
    if is_grasped:
        # Reward for getting cube A close to the goal
        if cubeA_to_goal_dist < 0.1:
            reward += 10.0

    # Stage 5: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Reward for releasing cube A at the goal
        gripper_openness = self.robot.gripper_openness
        if gripper_openness > 0.9:  # Encourage fully open gripper
            reward += 20.0

    # Penalize high velocity to ensure smooth movement
    velocity_penalty = -0.1 * np.linalg.norm(self.robot.qvel)
    reward += velocity_penalty

    # Penalize high angular velocity to ensure smooth movement
    angular_velocity_penalty = -0.1 * np.linalg.norm(self.cubeA.angular_velocity)
    reward += angular_velocity_penalty

    return reward