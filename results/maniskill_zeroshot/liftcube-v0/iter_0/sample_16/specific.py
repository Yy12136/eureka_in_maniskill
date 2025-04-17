import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.4     # Primary weight for lifting the cube
    weight_steady = 0.2   # Secondary weight for maintaining stability during the lift
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successfully grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube to the desired height
    reward_steady = 0.0   # Reward for maintaining stability during the lift
    
    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Measure the height difference between the cube and the goal height
    goal_height = 0.2
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube's position
    height_diff = max(0, goal_height - cube_height)
    reward_lift = 1.0 - (height_diff / goal_height)
    
    # 3. Steady reward: Penalize large velocity changes to ensure smooth lifting
    cube_velocity = self.obj.get_velocity()
    velocity_magnitude = (cube_velocity[0]**2 + cube_velocity[1]**2 + cube_velocity[2]**2) ** 0.5
    reward_steady = 1.0 - min(1.0, velocity_magnitude / 0.1)  # Normalize velocity magnitude
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= goal_height:
        reward += 0.1
    
    # 2. Penalty for large actions (regularization)
    action_magnitude = (action[0]**2 + action[1]**2 + action[2]**2) ** 0.5
    reward -= 0.05 * min(1.0, action_magnitude / 0.5)  # Normalize action magnitude
    
    return reward