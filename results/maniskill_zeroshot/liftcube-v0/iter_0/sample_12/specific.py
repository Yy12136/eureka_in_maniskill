import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_reach = 0.4    # Weight for reaching the cube
    weight_grasp = 0.3    # Weight for successfully grasping the cube
    weight_lift = 0.3    # Weight for lifting the cube to the desired height
    
    # Initialize reward components
    reward_reach = 0.0    # Reward for reaching the cube
    reward_grasp = 0.0    # Reward for grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube
    
    # Calculate reward components
    # Reward for reaching the cube (distance between end-effector and cube)
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    dist_to_cube = ((tcp_pos[0] - cube_pos[0])**2 + 
                    (tcp_pos[1] - cube_pos[1])**2 + 
                    (tcp_pos[2] - cube_pos[2])**2)**0.5
    reward_reach = max(0, 1 - dist_to_cube / 0.1)  # Normalize to [0, 1] within 0.1m
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube to the desired height
    if self.agent.check_grasp(self.obj):
        current_height = cube_pos[2]
        target_height = 0.2
        reward_lift = max(0, 1 - abs(current_height - target_height) / 0.1)  # Normalize to [0, 1] within 0.1m
    
    # Combine main rewards
    reward = (
        weight_reach * reward_reach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Additional reward components
    # Penalty for large actions (regularization)
    action_penalty = -0.01 * sum(a**2 for a in action)
    reward += action_penalty
    
    return reward