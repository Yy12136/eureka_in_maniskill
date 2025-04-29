import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_distance_obj = 0.8000    # Primary weight for object distance to goal
    weight_grasp = 0.1500    # Secondary weight for successful grasp
    weight_stability = 0.0500    # Additional weight for object stability
    
    # Initialize reward components (total number <= 5)
    reward_distance_obj = 0.0    # Main reward component for object distance
    reward_grasp = 0.0           # Main reward component for grasp success
    reward_stability = 0.0       # Main reward component for object stability
    
    # Calculate reward components
    # 1. Reward for reducing the distance between the object and the goal
    obj_to_goal_dist = np.linalg.norm(self.obj.pose.p - self.goal_pos)
    reward_distance_obj = 1.0 / (1.0 + obj_to_goal_dist)  # Inverse distance reward
    
    # 2. Reward for successful grasp of the object
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for object stability (minimizing object velocity)
    obj_velocity = np.linalg.norm(self.obj.velocity)
    reward_stability = 1.0 / (1.0 + obj_velocity)  # Inverse velocity reward
    
    # Combine main rewards
    reward = (
        weight_distance_obj * reward_distance_obj +
        weight_grasp * reward_grasp +
        weight_stability * reward_stability
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.obj.pose.p[2] >= self.goal_pos[2]:
        reward += 0.1  # Small bonus for keeping the cube above the goal height
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)  # Penalize large actions
    reward += action_penalty
    
    return reward