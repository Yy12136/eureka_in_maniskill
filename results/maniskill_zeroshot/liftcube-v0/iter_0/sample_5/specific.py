import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.5     # Primary weight for lifting the cube
    weight_smooth = 0.1   # Secondary weight for smooth motion
    # Note: weight_grasp + weight_lift + weight_smooth = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_smooth = 0.0   # Reward for smooth motion
    
    # Calculate reward components
    # 1. Grasping reward
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lifting reward
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2               # Target height in meters
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0, 1 - height_diff / target_height)  # Normalized reward
    
    # 3. Motion smoothness reward
    action_magnitude = sum([a**2 for a in action]) ** 0.5  # L2 norm of action
    reward_smooth = max(0, 1 - action_magnitude / 2.0)     # Normalized reward
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    if action_magnitude > 2.0:
        reward -= 0.05  # Small penalty for large actions
    
    return reward