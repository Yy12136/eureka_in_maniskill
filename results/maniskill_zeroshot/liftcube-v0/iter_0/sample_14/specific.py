import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasp
    weight_lift = 0.4     # Weight for lifting the cube
    weight_steady = 0.2   # Weight for smooth and steady motion
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0    # Reward for lifting the cube
    reward_steady = 0.0  # Reward for smooth and steady motion
    
    # Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate the height difference between the cube and the goal height
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    goal_height = 0.2  # Target height
    height_diff = max(0, goal_height - cube_height)
    reward_lift = 1.0 - (height_diff / goal_height)
    
    # Calculate smoothness of motion (penalize large actions)
    action_magnitude = sum([abs(a) for a in action])
    reward_steady = 1.0 - min(1.0, action_magnitude / 10.0)  # Normalize action magnitude
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Bonus for maintaining cube above goal height
    if cube_height >= goal_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # Optional: Penalty for large actions (regularization)
    reward -= 0.05 * action_magnitude  # Small penalty for large actions
    
    return reward