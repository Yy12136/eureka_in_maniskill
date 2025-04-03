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

    # Success reward
    if success:
        reward += 300.0  # Large reward for completing the task
        return reward

    # Stage 1: Grasping Cube A
    if not is_grasped:
        # Small reward for successfully grasping the cube
        if is_grasped:
            reward += 50.0

    # Stage 2: Moving Cube A to Goal
    if is_grasped:
        # Small reward for reaching the goal with the cube
        if is_cubeA_at_goal:
            reward += 100.0

    # Penalize dropping the cube
    if is_grasped and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward -= 100.0

    # Penalize excessive action magnitude to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward