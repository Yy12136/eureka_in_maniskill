import numpy as np

def compute_sparse_reward(self, action):
    reward = 0.0

    # Check if the drawer is opened beyond the target qpos
    is_drawer_opened = self.cabinet_drawer.qpos >= self.target_qpos

    # Check if the robot is static (optional, depending on task requirements)
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    # Task success condition
    success = is_drawer_opened and is_robot_static

    # Sparse reward for task completion
    if success:
        reward += 10.0  # Large reward for successful completion
        return reward

    # Optional: Small penalty for each step to encourage efficiency
    reward -= 0.01

    return reward