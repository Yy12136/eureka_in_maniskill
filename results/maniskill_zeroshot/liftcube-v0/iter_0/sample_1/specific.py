import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4750    # Weight for successful grasp
    weight_lift = 0.4750    # Weight for lifting the cube
    weight_steady = 0.0500    # Weight for steady motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_steady = 0.0   # Reward for steady motion
    
    # Calculate reward components
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube by 0.2 meters
    target_height = 0.2
    current_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    if current_height >= target_height:
        reward_lift = 1.0
    else:
        reward_lift = current_height / target_height
    
    # Reward for steady motion (penalize large actions)
    action_magnitude = sum([a**2 for a in action])**0.5
    reward_steady = 1.0 / (1.0 + action_magnitude)
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # Bonus for maintaining cube above goal height
    if current_height >= target_height:
        reward += 0.1
    
    # Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude
    
    return reward