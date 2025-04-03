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

    # Success reward
    if success:
        reward += 300.0  # Large reward for completing the task
        return reward

    # Stage 1: Grasping Cube A
    if not is_grasped:
        # Reward for successful grasp
        if is_grasped:
            reward += 50.0  # Reward for grasping the cube
        else:
            # Penalize unsuccessful grasp attempts
            reward -= 10.0  # Penalty for failing to grasp

    # Stage 2: Lifting Cube A
    if is_grasped:
        # Encourage lifting cube A to a certain height
        lifting_height = cubeA_pos[2] - self.cube_half_size
        if lifting_height > 0.1:  # Minimum height to consider it lifted
            reward += 50.0  # Reward for lifting the cube

        # Penalize if cube A is dropped
        if not is_grasped:
            reward -= 50.0  # Penalty for dropping the cube

    # Stage 3: Moving Cube A to Goal
    if is_grasped:
        # Reward for placing the cube at the goal
        if cubeA_to_goal_dist < 0.05:  # Close enough to the goal
            reward += 100.0  # Reward for placing the cube near the goal

        # Penalize if cube A moves away from the goal
        if cubeA_to_goal_dist > 0.5:
            reward -= 50.0  # Penalty for moving away from the goal

    # Stage 4: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Encourage gripper openness for releasing
        gripper_openness = self.robot.gripper_openness
        if gripper_openness > 0.9:  # Fully open gripper
            reward += 50.0  # Reward for releasing the cube

        # Penalize releasing the cube too early or too late
        if gripper_openness < 0.9:
            reward -= 50.0  # Penalty for incorrect gripper openness

    # Regularization of the robot's action
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    # Penalize high velocity to ensure smooth movement
    velocity_penalty = -0.2 * np.linalg.norm(self.robot.qvel)
    reward += velocity_penalty

    # Penalize high angular velocity to ensure smooth movement
    angular_velocity_penalty = -0.2 * np.linalg.norm(self.cubeA.angular_velocity)
    reward += angular_velocity_penalty

    return reward