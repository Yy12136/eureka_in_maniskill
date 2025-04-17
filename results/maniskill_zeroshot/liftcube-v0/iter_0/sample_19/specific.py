import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasping
    weight_lift = 0.4     # Weight for lifting the cube
    weight_smooth = 0.2   # Weight for smooth motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasping
    reward_lift = 0.0     # Reward for lifting the cube
    reward_smooth = 0.0   # Reward for smooth motion
    
    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Check if the cube is lifted by 0.2 meters
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    initial_height = 0.0  # Assuming initial height is 0
    target_height = initial_height + 0.2
    height_diff = cube_height - initial_height
    if height_diff >= 0.2:
        reward_lift = 1.0
    else:
        reward_lift = height_diff / 0.2
    
    # 3. Smooth motion reward: Penalize large actions
    action_magnitude = sum([abs(a) for a in action])
    reward_smooth = 1.0 - min(action_magnitude / 10.0, 1.0)
    
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