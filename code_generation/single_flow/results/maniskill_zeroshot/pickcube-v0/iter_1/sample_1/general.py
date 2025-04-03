import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Cube A
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] > 0.02:  # Z-coordinate of cube A
        reward += 1.0  # Reward for lifting the cube

    # Step 3: Move Cube A to Goal Position
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.1:
        reward += 1.0  # Reward for moving the cube close to the goal

    # Step 4: Release at Goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing the cube at the goal position

    # Penalties
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 0.5  # Penalize moving cube A away from the goal while grasping

    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0  # Penalize dropping cube A away from the goal

    return reward