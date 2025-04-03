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

    # Success reward (main focus)
    if success:
        reward += 300.0  # Large reward for completing the task
        return reward

    # Milestone 1: Reaching Cube A
    if not is_grasped and gripper_to_cubeA_dist < 0.05:  # Tolerance for reaching
        reward += 50.0  # Reward for reaching the cube

    # Milestone 2: Grasping Cube A
    if is_grasped:
        reward += 50.0  # Reward for successfully grasping the cube

    # Milestone 3: Moving Cube A to Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:  # Tolerance for proximity to goal
        reward += 50.0  # Reward for bringing the cube close to the goal

    # Penalty for dropping the cube
    if not is_grasped and self.cubeA.pose.p[2] > self.cube_half_size:  # Check if cube is lifted
        reward -= 100.0  # Large penalty for dropping the cube

    # Penalty for moving the cube away from the goal
    if is_grasped and cubeA_to_goal_dist > 0.5:  # Threshold for moving away
        reward -= 50.0  # Penalty for moving the cube away from the goal

    return reward