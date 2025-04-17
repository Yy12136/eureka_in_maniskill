import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.4     # Secondary weight for lifting the cube
    weight_steady = 0.2   # Additional weight for maintaining stability during lift
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successfully grasping the cube
    reward_lift = 0.0    # Reward for lifting the cube to the target height
    reward_steady = 0.0   # Reward for maintaining stability during the lift
    
    # Calculate reward components
    # 1. Grasp reward: Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Lift reward: Measure how close the cube is to the target height (0.2 meters)
    target_height = 0.2
    current_height = self.obj.pose.p[2]  # Z-coordinate of the cube's position
    height_diff = abs(current_height - target_height)
    reward_lift = max(0, 1 - height_diff / target_height)  # Normalized reward
    
    # 3. Steady reward: Penalize large velocity changes during the lift
    if self.agent.check_grasp(self.obj):
        qvel = self.agent.robot.get_qvel()[:-2]  # Exclude gripper joints
        velocity_magnitude = sum(abs(v) for v in qvel)
        reward_steady = max(0, 1 - velocity_magnitude / 10.0)  # Normalized reward
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_steady * reward_steady
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if current_height >= target_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    action_magnitude = sum(abs(a) for a in action)
    reward -= 0.05 * action_magnitude  # Penalize large actions
    
    return reward