import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Cube A above a certain height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] > 0.1:
        reward += 1.0  # Reward for lifting Cube A

    # Step 3: Move Cube A close to the goal
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.1:
        reward += 1.0  # Reward for moving Cube A near the goal

    # Step 4: Release Cube A at the goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing Cube A at the goal

    # Penalties
    # Penalty for dropping Cube A away from the goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0  # Penalty for dropping Cube A

    return reward