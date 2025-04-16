import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_reach = 0.4    # Weight for reaching the cube
    weight_grasp = 0.3    # Weight for successfully grasping the cube
    weight_lift = 0.3     # Weight for lifting the cube to the target height
    # Note: weight_reach + weight_grasp + weight_lift = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_reach = 0.0    # Reward for reaching the cube
    reward_grasp = 0.0    # Reward for grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube to the target height
    
    # Calculate reward components
    # 1. Reward for reaching the cube (distance between end-effector and cube)
    ee_pos = self.tcp.pose.p  # End-effector position
    cube_pos = self.obj.pose.p  # Cube position
    dist_to_cube = max(0, np.linalg.norm(ee_pos - cube_pos) - 0.02)
    reward_reach = 1.0 - np.tanh(5.0 * dist_to_cube)  # Scale distance reward
    
    # 2. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Reward for lifting the cube to the target height
    if self.agent.check_grasp(self.obj):
        target_height = cube_pos[2] + 0.2  # Target height is 0.2 meters above initial position
        current_height = self.obj.pose.p[2]
        height_diff = max(0, target_height - current_height)
        reward_lift = 1.0 - np.tanh(5.0 * height_diff)  # Scale height reward
    
    # Combine main rewards
    reward = (
        weight_reach * reward_reach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward