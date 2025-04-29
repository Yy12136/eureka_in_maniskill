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
    # Reward for successful grasping
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube to the target height
    target_height = 0.2
    current_height = self.obj.pose.p[2]
    if reward_grasp > 0.0:  # Only consider lifting if the cube is grasped
        reward_lift = max(0.0, 1.0 - abs(current_height - target_height) / target_height)
    
    # Reward for smooth motion (penalize large actions)
    action_magnitude = sum([abs(a) for a in action])
    reward_smooth = max(0.0, 1.0 - action_magnitude / len(action))
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )
    
    # Optional: Additional reward components
    # Bonus for maintaining cube above goal height
    if current_height >= target_height:
        reward += 0.1
    
    # Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude
    
    return reward