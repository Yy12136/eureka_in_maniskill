import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasp
    weight_lift = 0.4     # Weight for lifting the cube
    weight_steady = 0.2   # Weight for steady motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_steady = 0.0   # Reward for steady motion
    
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
        reward_lift = cube_height / target_height  # Linear interpolation
    
    # 3. Reward for steady motion (low velocity)
    cube_velocity = self.obj.get_velocity()
    velocity_magnitude = (cube_velocity[0]**2 + cube_velocity[1]**2 + cube_velocity[2]**2)**0.5
    reward_steady = max(0.0, 1.0 - velocity_magnitude / 0.1)  # Normalize to [0, 1]
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    action_magnitude = (action[0]**2 + action[1]**2 + action[2]**2)**0.5
    reward -= 0.05 * action_magnitude  # Penalize large actions
    
    return reward