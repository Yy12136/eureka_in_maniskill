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
    
    # Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        current_height = cube_pos[2]
        reward_lift = max(0, 1 - abs(current_height - target_height) / 0.1)  # Normalize to [0, 1] within 0.1m
    
    # Combine main rewards
    reward = (
        weight_reach * reward_reach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Penalty for large actions (regularization)
    action_penalty = -0.01 * sum([a**2 for a in action])  # Small penalty for large actions
    reward += action_penalty
    
    return reward