import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 2: Lift Cube A above a threshold height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] > 0.05:  # Z-coordinate threshold
        reward += 1.0  # Reward for lifting the cube

    # Step 3: Place Cube A at the goal position
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for placing the cube at the goal

    # Step 4: Task Completion Bonus
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 3.0  # Large bonus for completing the task

    return reward