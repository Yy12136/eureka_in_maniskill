import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4750    # Weight for successful grasp
    weight_lift = 0.4750    # Weight for lifting the cube
    weight_steady = 0.0500    # Weight for steady motion during lifting
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_steady = 0.0   # Reward for steady motion during lifting
    
    # Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate the height difference between the cube and the goal height (0.2 meters)
    cube_height = self.obj.pose.p[2]
    goal_height = 0.2
    height_diff = abs(cube_height - goal_height)
    
    # Reward for lifting the cube closer to the goal height
    if height_diff < 0.2:
        reward_lift = 1.0 - (height_diff / 0.2)
    
    # Reward for steady motion during lifting (low velocity of the cube)
    cube_velocity = self.obj.get_velocity()[2]
    if abs(cube_velocity) < 0.01:
        reward_steady = 1.0
    
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
    action_penalty = -0.01 * sum(abs(action))
    reward += action_penalty
    
    return reward