import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_approach = 0.4    # Weight for approaching the cube
    weight_grasp = 0.3      # Weight for successfully grasping the cube
    weight_lift = 0.3       # Weight for lifting the cube to the desired height
    # Note: weight_approach + weight_grasp + weight_lift = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_approach = 0.0    # Reward for approaching the cube
    reward_grasp = 0.0       # Reward for successfully grasping the cube
    reward_lift = 0.0        # Reward for lifting the cube to the desired height
    
    # Calculate reward components
    # 1. Reward for approaching the cube
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    dist_to_cube = max(0, ((tcp_pos[0] - cube_pos[0])**2 + (tcp_pos[1] - cube_pos[1])**2 + (tcp_pos[2] - cube_pos[2])**2)**0.5 - 0.02)
    reward_approach = max(0, 1 - dist_to_cube / 0.1)  # Normalize distance to [0, 1]
    
    # 2. Reward for successfully grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for lifting the cube to the desired height
    if self.agent.check_grasp(self.obj):
        current_height = cube_pos[2]
        target_height = 0.2
        height_diff = max(0, target_height - current_height)
        reward_lift = max(0, 1 - height_diff / 0.2)  # Normalize height difference to [0, 1]
    
    # Combine main rewards
    reward = (
        weight_approach * reward_approach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and cube_pos[2] >= 0.2:
        reward += 0.1
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * sum([a**2 for a in action])
    reward += action_penalty
    
    return reward