import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successful grasp
    weight_lift = 0.4     # Weight for lifting the cube
    weight_steady = 0.2   # Weight for steady motion during lifting
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_steady = 0.0   # Reward for steady motion during lifting
    
    # Calculate reward components
    # 1. Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Reward for lifting the cube by 0.2 meters
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
        reward_lift = min(max(cube_height / 0.2, 0.0), 1.0)  # Normalized to [0, 1]
    
    # 3. Reward for steady motion during lifting
    if self.agent.check_grasp(self.obj):
        cube_velocity = self.obj.get_velocity()  # Velocity of the cube
        reward_steady = 1.0 - min(np.linalg.norm(cube_velocity) / 0.1, 1.0)  # Penalize high velocity
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= 0.2:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward