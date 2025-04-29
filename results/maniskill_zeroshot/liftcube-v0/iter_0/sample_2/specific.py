import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_distance_ee = 0.4  # Weight for end-effector distance to cube
    weight_grasp = 0.3        # Weight for successful grasp
    weight_lift = 0.3         # Weight for lifting the cube to the target height

    # Initialize reward components
    reward_distance_ee = 0.0  # Reward for end-effector proximity to cube
    reward_grasp = 0.0        # Reward for successful grasp
    reward_lift = 0.0         # Reward for lifting the cube to the target height

    # Calculate reward components
    # 1. End-effector distance to cube
    ee_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    distance_ee_cube = np.linalg.norm(ee_pos - cube_pos)
    reward_distance_ee = max(0, 1 - distance_ee_cube / 0.1)  # Normalize distance to [0, 1]

    # 2. Successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0

    # 3. Lifting the cube to the target height
    target_height = 0.2
    if self.agent.check_grasp(self.obj):
        current_height = self.obj.pose.p[2]
        reward_lift = max(0, 1 - abs(current_height - target_height) / 0.1)  # Normalize height difference to [0, 1]

    # Combine main rewards
    reward = (
        weight_distance_ee * reward_distance_ee +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )

    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] >= target_height:
        reward += 0.1

    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward