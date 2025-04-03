import numpy as np

def compute_sparse_reward(self, action):
    reward = 0.0

    # Task completion: Chair is at the target location and upright
    chair_to_target_dist = np.linalg.norm(self.chair.pose.p[:2] - self.target_location[:2])
    chair_is_upright = np.abs(self.chair.pose.rot.euler_angles()[0]) < 0.2 and np.abs(self.chair.pose.rot.euler_angles()[1]) < 0.2
    chair_is_static = np.linalg.norm(self.chair.velocity) < 0.1 and np.linalg.norm(self.chair.angular_velocity) < 0.1

    if chair_to_target_dist < 0.1 and chair_is_upright and chair_is_static:
        reward += 10.0  # Large reward for task completion
        return reward

    # Penalize chair falling over
    if not chair_is_upright:
        reward -= 5.0  # Penalize heavily if the chair falls
        return reward

    # Progress toward the target location
    if chair_to_target_dist < self.prev_chair_to_target_dist:
        reward += 1.0  # Reward for making progress toward the target
    self.prev_chair_to_target_dist = chair_to_target_dist

    # Penalize excessive robot movement (to encourage efficiency)
    robot_movement_penalty = np.linalg.norm(action)
    reward -= 0.1 * robot_movement_penalty

    return reward