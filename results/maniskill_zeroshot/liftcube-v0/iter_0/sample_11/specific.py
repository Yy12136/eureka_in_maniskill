import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.4     # Primary weight for lifting the cube
    weight_smooth = 0.2   # Secondary weight for smooth motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successfully grasping the cube
    reward_lift = 0.0    # Reward for lifting the cube to the desired height
    reward_smooth = 0.0  # Reward for smooth motion during the task
    
    # Calculate reward components
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube to the desired height
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2  # Desired height in meters
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0.0, 1.0 - height_diff / target_height)
    
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
    if cube_height >= target_height:
        reward += 0.1
    
    # Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude
    
    return reward