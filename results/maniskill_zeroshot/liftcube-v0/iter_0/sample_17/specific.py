import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_reach = 0.4    # Weight for reaching the cube
    weight_grasp = 0.3    # Weight for successfully grasping the cube
    weight_lift = 0.3     # Weight for lifting the cube to the desired height
    # Note: weight_reach + weight_grasp + weight_lift = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_reach = 0.0    # Reward for reaching the cube
    reward_grasp = 0.0    # Reward for grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube
    
    # Calculate reward components
    # 1. Reward for reaching the cube
    tcp_to_cube_dist = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    reward_reach = max(0, 1 - tcp_to_cube_dist / 0.1)  # Normalize distance to [0, 1]
    
    # 2. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for lifting the cube to the desired height
    if self.agent.check_grasp(self.obj):
        cube_height = self.obj.pose.p[2]
        target_height = 0.2
        reward_lift = max(0, 1 - abs(cube_height - target_height) / 0.1)  # Normalize height difference to [0, 1]
    
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