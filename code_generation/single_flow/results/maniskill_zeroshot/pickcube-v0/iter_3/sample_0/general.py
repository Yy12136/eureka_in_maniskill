import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Key Milestones
    # Milestone 1: Grasp Cube A
    if not self.has_grasped and self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successfully grasping Cube A
        self.has_grasped = True  # Track that Cube A has been grasped

    # Milestone 2: Lift Cube A above a certain height
    if self.has_grasped and self.obj.pose.p[2] > 0.1:
        reward += 1.0  # Reward for lifting Cube A above 0.1 meters
        self.has_lifted = True  # Track that Cube A has been lifted

    # Milestone 3: Move Cube A close to the goal
    if self.has_lifted and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.1:
        reward += 1.0  # Reward for moving Cube A close to the goal
        self.is_near_goal = True  # Track that Cube A is near the goal

    # Milestone 4: Release Cube A at the goal
    if self.is_near_goal and not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing Cube A at the goal
        self.task_completed = True  # Track that the task is completed

    # Penalties
    # Penalty for dropping Cube A before reaching the goal
    if self.has_grasped and not self.agent.check_grasp(self.obj) and not self.task_completed:
        reward -= 1.0  # Penalty for dropping Cube A prematurely

    # Penalty for moving Cube A away from the goal while grasping
    if self.has_grasped and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.2:
        reward -= 0.5  # Penalty for moving Cube A away from the goal

    return reward