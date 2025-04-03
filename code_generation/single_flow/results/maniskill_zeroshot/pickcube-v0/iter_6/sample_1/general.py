import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Height
    if self.agent.check_grasp(self.obj):
        cubeA_height = self.obj.pose.p[2]  # Z-coordinate of cube A
        if cubeA_height > 0.1:  # Encourage lifting above a certain height
            reward += 1.0

    # Step 3: Release at Goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing at the goal position

    # Penalties
    # Penalty for dropping Cube A away from the goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0

    # Penalty for moving Cube A away from the goal while grasping
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 0.5

    return reward