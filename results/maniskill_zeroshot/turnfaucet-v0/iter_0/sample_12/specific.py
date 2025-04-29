import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_progress = 0.6    # Primary weight for task progress
    weight_grasp = 0.3       # Secondary weight for grasp state
    weight_control = 0.1     # Additional weight for motion control
    # Note: weight_progress + weight_grasp + weight_control = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_progress = 0.0    # Main reward component for task progress
    reward_grasp = 0.0       # Main reward component for grasp state
    reward_control = 0.0     # Main reward component for motion control
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the handle is to the target angle
    current_angle = self.current_angle
    target_angle = self.target_angle
    angle_diff = abs(target_angle - current_angle)
    reward_progress = 1.0 - (angle_diff / target_angle)  # Normalized progress
    
    # 2. Grasp State: Reward for maintaining a stable grasp on the handle
    if self.agent.check_grasp(self.target_link):
        reward_grasp = 1.0  # Full reward if the handle is grasped
    
    # 3. Motion Control: Penalize large actions to encourage smooth motion
    action_magnitude = sum([a**2 for a in action]) ** 0.5
    reward_control = 1.0 - min(action_magnitude, 1.0)  # Normalized control
    
    # Combine main rewards
    reward = (
        weight_progress * reward_progress +
        weight_grasp * reward_grasp +
        weight_control * reward_control
    )
    
    # Optional: Additional reward components
    # 1. Bonus for completing the task
    if current_angle >= target_angle:
        reward += 1.0  # Bonus for task completion
    
    # 2. Penalty for joint limits (optional)
    joint_positions = self.agent.robot.get_qpos()[:-2]
    joint_limits = [abs(pos) > 1.0 for pos in joint_positions]  # Example limit check
    if any(joint_limits):
        reward -= 0.1  # Penalty for exceeding joint limits
    
    return reward