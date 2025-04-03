import numpy as np

def compute_sparse_reward(self, action):
    reward = 0.0

    # Task completion conditions
    is_chair_at_target = np.linalg.norm(self.chair.pose.p[:2] - self.target_position[:2]) <= 0.1  # Chair is within 0.1m of target in XY plane
    is_chair_upright = np.abs(self.chair.pose.rotation.to_euler("xyz")[0]) <= 0.2 and np.abs(self.chair.pose.rotation.to_euler("xyz")[1]) <= 0.2  # Chair is upright (roll and pitch angles within 0.2 radians)
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2  # Robot is static

    # Success condition
    success = is_chair_at_target and is_chair_upright and is_robot_static

    if success:
        reward += 10.0  # Large reward for task completion
        return reward

    # Sparse reward for progress toward the target
    if is_chair_upright:
        dist_to_target = np.linalg.norm(self.chair.pose.p[:2] - self.target_position[:2])
        if dist_to_target < 0.5:  # Reward for being close to the target
            reward += 2.0
        elif dist_to_target < 1.0:  # Reward for being moderately close to the target
            reward += 1.0

    # Penalty for chair falling over
    if not is_chair_upright:
        reward -= 5.0  # Large penalty for chair falling over

    return reward