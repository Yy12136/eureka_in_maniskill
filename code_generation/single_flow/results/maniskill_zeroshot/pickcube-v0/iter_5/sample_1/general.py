import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successfully grasping Cube A

    # Step 2: Lift Cube A above a threshold height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] > 0.1:
        reward += 1.0  # Reward for lifting Cube A above a certain height

    # Step 3: Move Cube A close to the goal position
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.1:
        reward += 1.0  # Reward for moving Cube A close to the goal

    # Step 4: Release Cube A at the goal position
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing Cube A at the goal position

    # Penalty for dropping Cube A away from the goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0  # Penalty for dropping Cube A away from the goal

    # Penalty for moving Cube A away from the goal while grasping
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 0.5  # Penalty for moving Cube A away from the goal while grasping

    return reward