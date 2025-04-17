import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasp
    weight_lift = 0.4     # Weight for lifting the cube
    weight_smooth = 0.2   # Weight for smooth motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_smooth = 0.0   # Reward for smooth motion
    
    # Calculate reward components
    # 1. Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Reward for lifting the cube by 0.2 meters
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2
    if cube_height >= target_height:
        reward_lift = 1.0
    else:
        reward_lift = cube_height / target_height
    
    # 3. Reward for smooth motion (penalize large actions)
    action_magnitude = sum([a**2 for a in action]) ** 0.5
    reward_smooth = 1.0 / (1.0 + action_magnitude)
    
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