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
    # Reward for approaching the cube
    tcp_to_cube_dist = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    reward_approach = max(0, 1 - tcp_to_cube_dist / 0.1)  # Normalize to [0, 1]
    
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        cube_height = self.obj.pose.p[2]
        target_height = 0.2
        reward_lift = max(0, 1 - abs(cube_height - target_height) / 0.1)  # Normalize to [0, 1]
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_approach * reward_approach
    )
    
    # Optional: Additional reward components
    # Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward