import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Height (Cube A must be lifted above a certain height)
    if self.agent.check_grasp(self.obj):
        cubeA_height = self.obj.pose.p[2]  # Z-coordinate of cube A
        if cubeA_height > 0.05:  # Encourage lifting above a threshold
            reward += 1.0

    # Step 3: Distance to Goal Position (Cube A must be close to the goal)
    if self.agent.check_grasp(self.obj):
        distance_to_goal = np.linalg.norm(self.obj.pose.p - self.goal_pos)
        if distance_to_goal < 0.05:  # Encourage proximity to the goal
            reward += 1.0

    # Step 4: Release at Goal (Cube A must be released at the goal position)
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing at the goal position

    # Penalties
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0  # Penalize dropping cube A away from the goal

    return reward