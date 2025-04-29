import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_task_progress = 0.6    # Primary weight for task progress
    weight_grasp = 0.2            # Secondary weight for grasp state
    weight_effort = 0.2           # Additional weight for motion control
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_grasp = 0.0            # Main reward component for grasp state
    reward_effort = 0.0           # Main reward component for motion control
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the handle is to the target angle
    angle_diff = self.target_angle - self.current_angle
    reward_task_progress = max(0.0, 1.0 - abs(angle_diff) / self.target_angle)
    
    # 2. Grasp State: Reward if the handle is grasped
    if self.agent.check_grasp(self.target_link):
        reward_grasp = 1.0
    
    # 3. Motion Control: Penalize large actions to encourage smooth motion
    action_magnitude = sum([a**2 for a in action]) ** 0.5
    reward_effort = max(0.0, 1.0 - action_magnitude / 10.0)
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp * reward_grasp +
        weight_effort * reward_effort
    )
    
    # Optional: Additional reward components
    # 1. Bonus for achieving the target angle
    if self.current_angle >= self.target_angle:
        reward += 1.0
    
    return reward