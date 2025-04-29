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
    
    # Calculate reward components
    # 1. Reward for reaching the cube
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    dist_to_cube = max(0, np.linalg.norm(tcp_pos - cube_pos) - 0.02)
    reward_reach = 1.0 / (1.0 + dist_to_cube)
    
    # 2. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        current_height = self.obj.pose.p[2]
        reward_lift = 1.0 - np.clip(abs(current_height - target_height) / target_height, 0, 1)
    
    # Combine main rewards
    reward = (
        weight_reach * reward_reach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] >= 0.2:
        reward += 0.1
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward