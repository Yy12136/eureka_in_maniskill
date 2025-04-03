import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get positions
    cubeA_pos = self.cubeA.pose.p
    goal_pos = self.goal_position

    # Calculate distance between cube A and the goal
    cubeA_to_goal_dist = np.linalg.norm(cubeA_pos - goal_pos)

    # Check if cube A is grasped
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)

    # Check if cube A is at the goal position and static
    is_cubeA_at_goal = cubeA_to_goal_dist < 0.01  # Tolerance for success
    is_cubeA_static = self.cubeA.check_static()
    success = is_cubeA_at_goal and is_cubeA_static and not is_grasped

    # Success reward (main focus)
    if success:
        reward += 100.0  # Large reward for task completion
        return reward

    # Stage 1: Grasping Cube A
    if not is_grasped:
        # Small reward for successful grasp
        if is_grasped:
            reward += 10.0

    # Stage 2: Moving Cube A to Goal
    if is_grasped:
        # Reward for reducing distance to the goal
        if cubeA_to_goal_dist < 0.1:  # Close to the goal
            reward += 5.0

    # Stage 3: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Reward for releasing the cube at the goal
        if not is_grasped:
            reward += 10.0

    # Penalize dropping the cube
    if not is_grasped and cubeA_to_goal_dist > 0.05:
        reward -= 10.0

    return reward