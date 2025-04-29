import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_task_progress = 0.6208    # Primary weight for task progress
    weight_grasp = 0.3292    # Secondary weight for grasp state
    weight_motion_control = 0.0500    # Additional weight for motion control
    weight_task_progress = 0.6208    # Note: weight_task_progress + weight_grasp + weight_motion_control = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_grasp = 0.0            # Main reward component for grasp state
    reward_motion_control = 0.0   # Main reward component for motion control
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the handle is to the target angle
    reward_task_progress = max(0.0, (self.current_angle - self.target_angle) / self.target_angle)
    
    # 2. Grasp State: Reward if the handle is grasped
    if self.agent.check_grasp(self.target_link):
        reward_grasp = 1.0
    
    # 3. Motion Control: Penalize large actions to encourage smooth motion
    action_magnitude = sum([a**2 for a in action])**0.5
    reward_motion_control = 1.0 / (1.0 + action_magnitude)
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp * reward_grasp +
        weight_motion_control * reward_motion_control
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    # 2. Penalty for large actions (regularization)
    
    return reward