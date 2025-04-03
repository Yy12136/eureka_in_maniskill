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

    # Success reward (large positive reward for task completion)
    if success:
        reward += 100.0
        return reward

    # Stage 1: Reaching Cube A
    if not is_grasped:
        # Small reward for getting close to cube A
        if gripper_to_cubeA_dist < 0.05:
            reward += 10.0

    # Stage 2: Grasping Cube A
    if not is_grasped:
        # Moderate reward for successful grasp
        if is_grasped:
            reward += 20.0

    # Stage 3: Moving Cube A to Goal
    if is_grasped:
        # Small reward for getting close to the goal
        if cubeA_to_goal_dist < 0.1:
            reward += 10.0

    # Stage 4: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Moderate reward for releasing cube A at the goal
        if not is_grasped:
            reward += 20.0

    # Regularization of the robot's action (small penalty to discourage large actions)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward