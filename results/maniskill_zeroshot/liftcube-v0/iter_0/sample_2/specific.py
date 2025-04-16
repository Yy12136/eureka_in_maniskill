import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Reward for successful grasp
    weight_lift = 0.4     # Reward for lifting the cube to the target height
    weight_smooth = 0.2   # Reward for smooth motion of the robot

    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Main reward component for grasping
    reward_lift = 0.0     # Main reward component for lifting
    reward_smooth = 0.0   # Main reward component for smooth motion

    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0

    # 2. Lift reward: Check if the cube is lifted to the target height
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0.0, 1.0 - height_diff / target_height)

    # 3. Smooth motion reward: Penalize large actions for smoothness
    action_magnitude = sum([abs(a) for a in action])
    reward_smooth = max(0.0, 1.0 - action_magnitude / len(action))

    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )

    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1

    # 2. Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude

    return reward