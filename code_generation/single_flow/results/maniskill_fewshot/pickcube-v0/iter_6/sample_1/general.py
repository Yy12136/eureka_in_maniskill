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

    # Check if cube A is at the goal position
    is_cubeA_at_goal = cubeA_to_goal_dist < 0.01  # Tolerance for success
    is_cubeA_static = self.cubeA.check_static()
    success = is_cubeA_at_goal and is_cubeA_static and not is_grasped

    # Success reward
    if success:
        reward += 1.0  # Sparse reward for completing the task
        return reward

    # Penalize dropping the cube
    if is_grasped and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward -= 0.5  # Penalty for dropping the cube

    # Regularization of the robot's action (optional, to encourage efficiency)
    action_penalty = -0.001 * np.linalg.norm(action)
    reward += action_penalty

    return reward