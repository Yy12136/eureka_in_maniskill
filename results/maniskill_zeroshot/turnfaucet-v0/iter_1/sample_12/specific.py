import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_task_progress = 0.7    # Primary weight for task progress
    weight_grasp_success = 0.3    # Secondary weight for successful grasp
    # Note: weight_task_progress + weight_grasp_success = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_grasp_success = 0.0    # Main reward component for successful grasp
    
    # Calculate reward components
    # 1. Task progress: reward based on how close the current angle is to the target angle
    angle_diff = self.target_angle - self.current_angle
    reward_task_progress = max(0.0, 1.0 - abs(angle_diff) / self.target_angle)
    
    # 2. Grasp success: reward if the handle is successfully grasped
    if self.agent.check_grasp(self.target_link):
        reward_grasp_success = 1.0
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp_success * reward_grasp_success
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    # 2. Penalty for large actions (regularization)
    
    return reward