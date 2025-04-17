import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for successful grasp
    weight_lift = 0.4     # Secondary weight for lifting the cube
    weight_steady = 0.2   # Additional weight for maintaining stability
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Main reward component for grasping
    reward_lift = 0.0     # Main reward component for lifting
    reward_steady = 0.0   # Main reward component for stability
    
    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Measure how close the cube is to the target height (0.2 meters)
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0, 1 - height_diff / target_height)  # Normalized reward
    
    # 3. Steady reward: Check if the cube is stable (not moving)
    if check_actor_static(self.obj):
        reward_steady = 1.0
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1  # Small bonus for achieving the goal
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * sum(a**2 for a in action)  # L2 regularization
    reward += action_penalty
    
    return reward