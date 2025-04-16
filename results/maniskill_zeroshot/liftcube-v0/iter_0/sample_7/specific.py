import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_reach = 0.4    # Weight for reaching the cube
    weight_grasp = 0.3    # Weight for grasping the cube
    weight_lift = 0.3     # Weight for lifting the cube
    # Note: weight_reach + weight_grasp + weight_lift = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_reach = 0.0    # Reward for reaching the cube
    reward_grasp = 0.0    # Reward for grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube
    
    # Get the position of the cube and the end-effector
    cube_pos = self.obj.pose.p
    ee_pos = self.tcp.pose.p
    
    # Calculate the distance between the end-effector and the cube
    dist_to_cube = ((ee_pos[0] - cube_pos[0])**2 + 
                    (ee_pos[1] - cube_pos[1])**2 + 
                    (ee_pos[2] - cube_pos[2])**2)**0.5
    
    # Reward for reaching the cube (inverse of distance)
    reward_reach = max(0, 1 - dist_to_cube / 0.1)  # Normalize to [0, 1] within 0.1m
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube by 0.2 meters
    if self.agent.check_grasp(self.obj):
        lift_height = cube_pos[2] - 0.02  # Subtract half size to get the base
        reward_lift = max(0, min(1, lift_height / 0.2))  # Normalize to [0, 1] for 0.2m
    
    # Combine main rewards
    reward = (
        weight_reach * reward_reach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and cube_pos[2] - 0.02 >= 0.2:
        reward += 0.1  # Small bonus for maintaining the lift
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * sum(a**2 for a in action)  # Penalize large actions
    reward += action_penalty
    
    return reward