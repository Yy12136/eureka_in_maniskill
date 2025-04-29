import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.4     # Primary weight for lifting the cube
    weight_steady = 0.2   # Secondary weight for maintaining stability during the lift
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successfully grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube to the target height
    reward_steady = 0.0   # Reward for maintaining stability during the lift
    
    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Measure how close the cube is to the target height (0.2 meters)
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube's position
    target_height = 0.2
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0, 1 - height_diff / target_height)  # Normalized reward
    
    # 3. Steady reward: Penalize large velocity of the cube during the lift
    cube_velocity = self.obj.get_velocity()[2]  # Z-component of velocity
    reward_steady = max(0, 1 - abs(cube_velocity) / 0.1)  # Normalized reward
    
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