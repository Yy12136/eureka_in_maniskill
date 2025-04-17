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
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2
    height_diff = max(0, target_height - cube_height)
    reward_lift = 1.0 - (height_diff / target_height)
    
    # Reward for smooth motion (penalize large actions)
    action_magnitude = sum([a**2 for a in action])**0.5
    max_action_magnitude = 1.0  # Assuming normalized actions
    reward_smooth = 1.0 - (action_magnitude / max_action_magnitude)
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )
    
    # Optional: Additional reward components
    # Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1
    
    # Penalty for large actions (regularization)
    if action_magnitude > max_action_magnitude:
        reward -= 0.1
    
    return reward