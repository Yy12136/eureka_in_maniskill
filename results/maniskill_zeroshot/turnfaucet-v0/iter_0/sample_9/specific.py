import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_progress = 0.6    # Primary weight for task progress
    weight_grasp = 0.2       # Secondary weight for grasp state
    weight_control = 0.2    # Additional weight for motion control
    # Note: weight_progress + weight_grasp + weight_control = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_progress = 0.0    # Main reward component for task progress
    reward_grasp = 0.0       # Main reward component for grasp state
    reward_control = 0.0     # Main reward component for motion control
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the handle is to the target angle
    progress = (self.current_angle - self.target_angle) / self.target_angle
    reward_progress = max(0.0, 1.0 - abs(progress))
    
    # 2. Grasp State: Reward for maintaining a stable grasp on the handle
    if self.agent.check_grasp(self.target_link):
        reward_grasp = 1.0
    else:
        reward_grasp = 0.0
    
    # 3. Motion Control: Reward for smooth and controlled motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward_control = 1.0 - min(1.0, np.linalg.norm(joint_velocities) / 10.0)
    
    # Combine main rewards
    reward = (
        weight_progress * reward_progress +
        weight_grasp * reward_grasp +
        weight_control * reward_control
    )
    
    # Optional: Additional reward components
    # 1. Bonus for achieving the target angle
    if self.current_angle >= self.target_angle:
        reward += 1.0
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward