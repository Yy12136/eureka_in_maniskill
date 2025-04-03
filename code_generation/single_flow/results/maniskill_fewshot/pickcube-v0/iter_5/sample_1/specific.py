import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
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
        reward += 250.0  # Increased success reward for completing the task
        return reward

    # Stage 1: Reaching Cube A
    if not is_grasped:
        # Reward for reducing distance between gripper and cube A
        reaching_reward = 1 - np.tanh(5.0 * gripper_to_cubeA_dist)
        reward += reaching_reward

        # Encourage gripper openness for grasping
        gripper_openness = self.robot.gripper_openness
        gripper_openness_reward = 1 - np.abs(gripper_openness - 0.5)  # Encourage gripper to be half-open
        reward += 4.5 * gripper_openness_reward  # Increased weight for gripper openness

        # Penalize excessive gripper openness
        if gripper_openness > 0.7:
            reward -= 3.5  # Increased penalty for excessive openness

    # Stage 2: Grasping Cube A
    if not is_grasped:
        # Additional reward for successful grasp
        if is_grasped:
            reward += 35.0  # Increased reward for successful grasp
        else:
            # Penalize unsuccessful grasp attempts
            reward -= 12.0  # Increased penalty for unsuccessful grasp attempts

    # Stage 3: Lifting Cube A
    if is_grasped:
        # Encourage lifting cube A to a certain height
        lifting_height = cubeA_pos[2] - self.cube_half_size
        lifting_reward = min(lifting_height / 0.2, 1.0)  # Target height is 0.2 meters
        reward += lifting_reward

        # Penalize if cube A is dropped
        if not is_grasped:
            reward -= 45.0  # Increased penalty for dropping the cube

        # Reward maintaining a stable grasp
        if is_grasped:
            reward += 6.0  # Increased reward for maintaining a stable grasp

    # Stage 4: Moving Cube A to Goal
    if is_grasped:
        # Encourage moving cube A toward the goal
        moving_reward = 1 - np.tanh(5.0 * cubeA_to_goal_dist)
        reward += moving_reward

        # Penalize if cube A moves away from the goal
        if cubeA_to_goal_dist > 0.5:
            reward -= 35.0  # Increased penalty for moving away from the goal

        # Reward maintaining a stable grasp while moving
        if is_grasped:
            reward += 6.0  # Increased reward for maintaining a stable grasp while moving

    # Stage 5: Releasing Cube A at Goal
    if is_grasped and cubeA_to_goal_dist < 0.05:
        # Encourage gripper openness for releasing
        gripper_openness = self.robot.gripper_openness
        gripper_openness_reward = 1 - np.abs(gripper_openness - 1.0)  # Encourage fully open gripper
        reward += 35.0 * gripper_openness_reward  # Increased reward for correct gripper openness

        # Penalize releasing the cube too early or too late
        if gripper_openness < 0.9:
            reward -= 30.0  # Increased penalty for incorrect gripper openness

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