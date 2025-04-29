import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_task_progress = 0.6    # Primary weight for task progress
    weight_grasp_success = 0.3    # Secondary weight for successful grasp
    weight_motion_quality = 0.1   # Additional weight for smooth motion
    # Note: weight_task_progress + weight_grasp_success + weight_motion_quality = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_grasp_success = 0.0    # Main reward component for successful grasp
    reward_motion_quality = 0.0   # Main reward component for smooth motion
    
    # Calculate reward components
    # 1. Task progress: reward based on how close the current angle is to the target angle
    angle_diff = self.target_angle - self.current_angle
    reward_task_progress = max(0.0, 1.0 - abs(angle_diff) / self.target_angle)
    
    # 2. Grasp success: reward if the handle is successfully grasped
    if self.agent.check_grasp(self.target_link):
        reward_grasp_success = 1.0
    
    # 3. Motion quality: penalize jerky movements to encourage smooth motion
    action_magnitude = sum([a**2 for a in action])**0.5
    reward_motion_quality = max(0.0, 1.0 - action_magnitude / 10.0)
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp_success * reward_grasp_success +
        weight_motion_quality * reward_motion_quality
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining a stable grasp throughout the task
    # 2. Penalty for large actions (regularization)
    
    return reward