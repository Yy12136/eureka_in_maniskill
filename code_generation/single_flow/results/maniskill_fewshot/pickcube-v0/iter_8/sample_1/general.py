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

    # Success reward (primary focus)
    if success:
        reward += 300.0  # Large reward for completing the task
        return reward

    # Stage 1: Reaching Cube A
    if not is_grasped and gripper_to_cubeA_dist < 0.05:
        reward += 10.0  # Small reward for getting close to the cube

    # Stage 2: Grasping Cube A
    if is_grasped:
        reward += 20.0  # Reward for successful grasp

    # Stage 3: Lifting Cube A
    if is_grasped and cubeA_pos[2] - self.cube_half_size > 0.1:
        reward += 15.0  # Reward for lifting the cube to a certain height

    # Stage 4: Moving Cube A to Goal
    if is_grasped and cubeA_to_goal_dist < 0.1:
        reward += 25.0  # Reward for bringing the cube close to the goal

    # Stage 5: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05 and self.robot.gripper_openness > 0.9:
        reward += 30.0  # Reward for releasing the cube at the goal

    # Penalties for undesirable behaviors
    if not is_grasped and gripper_to_cubeA_dist > 0.1:
        reward -= 5.0  # Penalty for being too far from the cube

    if is_grasped and cubeA_to_goal_dist > 0.5:
        reward -= 10.0  # Penalty for moving the cube away from the goal

    if not is_grasped and cubeA_pos[2] - self.cube_half_size > 0.1:
        reward -= 15.0  # Penalty for lifting the cube without grasping

    # Regularization of the robot's action
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward