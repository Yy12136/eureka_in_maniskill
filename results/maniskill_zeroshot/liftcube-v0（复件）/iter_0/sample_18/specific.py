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
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Measure the height of the cube relative to the goal
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    goal_height = 0.2  # Target height in meters
    height_diff = max(0, goal_height - cube_height)  # Ensure non-negative
    reward_lift = 1.0 - (height_diff / goal_height)  # Normalized reward
    
    # 3. Smooth motion reward: Penalize large actions for smoothness
    action_magnitude = sum(a**2 for a in action) ** 0.5  # L2 norm of action
    reward_smooth = 1.0 / (1.0 + action_magnitude)  # Inverse scaling
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= goal_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude  # Small penalty for large actions
    
    return reward