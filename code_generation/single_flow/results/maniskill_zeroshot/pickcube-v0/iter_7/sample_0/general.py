import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Cube A to a certain height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] > 0.1:
        reward += 1.0  # Reward for lifting Cube A above a certain height

    # Step 3: Place Cube A at the goal position
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for placing Cube A at the goal

    # Penalty for dropping Cube A
    if not self.agent.check_grasp(self.obj) and self.obj.pose.p[2] < 0.02:
        reward -= 1.0  # Penalty for dropping Cube A

    # Penalty for moving Cube A away from the goal while grasping
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 0.5  # Penalize moving Cube A away from the goal

    return reward