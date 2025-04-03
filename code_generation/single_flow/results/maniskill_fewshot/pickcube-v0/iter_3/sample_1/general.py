import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get positions
    cubeA_pos = self.cubeA.pose.p
    goal_pos = self.goal_position

    # Calculate distance
    cubeA_to_goal_dist = np.linalg.norm(cubeA_pos - goal_pos)

    # Check if cube A is grasped
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)

    # Check if cube A is at the goal position
    is_cubeA_at_goal = cubeA_to_goal_dist < 0.01  # Tolerance for success
    is_cubeA_static = self.cubeA.check_static()
    success = is_cubeA_at_goal and is_cubeA_static and not is_grasped

    # Success reward (primary focus)
    if success:
        reward += 100.0  # Large reward for task completion
        return reward

    # Grasping reward (secondary milestone)
    if is_grasped:
        reward += 20.0  # Moderate reward for grasping the cube

    # Moving toward goal reward (tertiary milestone)
    if is_grasped and cubeA_to_goal_dist < 0.5:
        reward += 10.0  # Small reward for moving the cube closer to the goal

    return reward