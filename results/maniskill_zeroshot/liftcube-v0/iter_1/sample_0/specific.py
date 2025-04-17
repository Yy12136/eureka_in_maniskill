import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Weight for successfully grasping the cube
    weight_lift = 0.4     # Weight for lifting the cube to the desired height
    weight_steady = 0.2   # Weight for maintaining stability during the lift
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for grasping the cube
    reward_lift = 0.0     # Reward for lifting the cube
    reward_steady = 0.0   # Reward for maintaining stability
    
    # Calculate reward components
    # 1. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Reward for lifting the cube to the desired height
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        cube_pos = self.obj.pose.p
        current_height = cube_pos[2]
        height_diff = max(0, target_height - current_height)
        reward_lift = 1.0 - np.tanh(10.0 * height_diff)  # Scale height difference to [0, 1]
    
    # 3. Reward for maintaining stability during the lift
    if self.agent.check_grasp(self.obj):
        # Check if the cube is not moving significantly (stable)
        cube_velocity = np.linalg.norm(self.obj.velocity)
        reward_steady = 1.0 - np.tanh(5.0 * cube_velocity)  # Scale velocity to [0, 1]
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if self.agent.check_grasp(self.obj) and cube_pos[2] >= target_height:
        reward += 0.1  # Small bonus for maintaining the height
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward