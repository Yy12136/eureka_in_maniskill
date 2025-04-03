import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get positions
    cubeA_pos = self.cubeA.pose.p
    goal_pos = self.goal_position

    # Calculate distance between cube A and goal
    cubeA_to_goal_dist = np.linalg.norm(cubeA_pos - goal_pos)

    # Check if cube A is grasped
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)

    # Check if cube A is at the goal position
    is_cubeA_at_goal = cubeA_to_goal_dist < 0.01  # Tolerance for success
    is_cubeA_static = self.cubeA.check_static()
    success = is_cubeA_at_goal and is_cubeA_static and not is_grasped

    # Success reward
    if success:
        reward += 100.0  # Large reward for task completion
        return reward

    # Penalize dropping the cube during the task
    if is_grasped and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward -= 50.0  # Large penalty for dropping the cube

    # Penalize moving the cube away from the goal significantly
    if is_grasped and cubeA_to_goal_dist > 0.5:
        reward -= 30.0  # Penalty for moving the cube too far from the goal

    # Regularization of the robot's action (optional, to encourage smoother actions)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward