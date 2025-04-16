import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_1 = 0.5    # Primary task weight: lifting the cube
    weight_2 = 0.3    # Secondary task weight: grasping the cube
    weight_3 = 0.2    # Additional weight: smooth motion
    # Note: weight_1 + weight_2 + weight_3 = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_1 = 0.0    # Main reward component: lifting the cube
    reward_2 = 0.0    # Main reward component: grasping the cube
    reward_3 = 0.0    # Main reward component: smooth motion
    
    # Calculate reward components
    # 1. Reward for lifting the cube by 0.2 meters
    target_height = 0.2
    current_height = self.obj.pose.p[2]  # Z-coordinate of cube's position
    reward_1 = max(0, 1 - abs(current_height - target_height) / target_height)
    
    # 2. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_2 = 1.0
    
    # 3. Reward for smooth motion (penalize large actions)
    action_magnitude = sum([a**2 for a in action]) ** 0.5
    reward_3 = max(0, 1 - action_magnitude / 10.0)  # Normalize by a reasonable max action magnitude
    
    # Combine main rewards
    reward = (
        weight_1 * reward_1 +
        weight_2 * reward_2 +
        weight_3 * reward_3
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if current_height >= target_height:
        reward += 0.1
    
    # 2. Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude
    
    return reward