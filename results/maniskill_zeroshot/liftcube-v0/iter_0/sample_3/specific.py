import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasp
    weight_lift = 0.4     # Weight for lifting the cube
    weight_approach = 0.2 # Weight for approaching the cube
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_approach = 0.0 # Reward for approaching the cube
    
    # Calculate reward components
    # 1. Reward for approaching the cube
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    dist_to_cube = max(0, np.linalg.norm(tcp_pos - cube_pos) - 0.02)
    reward_approach = 1.0 - np.tanh(5.0 * dist_to_cube)
    
    # 2. Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        current_height = self.obj.pose.p[2]
        height_diff = max(0, target_height - current_height)
        reward_lift = 1.0 - np.tanh(5.0 * height_diff)
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_approach * reward_approach
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and self.obj.pose.p[2] >= 0.2:
        reward += 0.1
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward