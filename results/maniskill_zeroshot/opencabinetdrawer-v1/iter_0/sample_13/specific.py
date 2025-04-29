import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5) and use "weight_" prefix for all weight variables.
    weight_task_progress = 0.6    # Primary weight for task progress
    weight_grasp_success = 0.2    # Secondary weight for successful grasp
    weight_motion_control = 0.2    # Additional weight for smooth motion control
    # Note: weight_task_progress + weight_grasp_success + weight_motion_control = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_grasp_success = 0.0    # Main reward component for successful grasp
    reward_motion_control = 0.0    # Main reward component for smooth motion control
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    progress = (self.link_qpos - self.target_qpos) / self.target_qpos
    reward_task_progress = max(0.0, min(1.0, progress))  # Normalize progress between 0 and 1
    
    # 2. Grasp Success: Reward if the end-effector is close to the handle
    handle_pos = self.target_link.pose.p
    ee_pos = self.agent.hand.pose.p
    distance_to_handle = np.linalg.norm(handle_pos - ee_pos)
    reward_grasp_success = 1.0 - min(1.0, distance_to_handle / 0.1)  # Normalize distance between 0 and 1
    
    # 3. Motion Control: Penalize large actions to encourage smooth motion
    action_magnitude = np.linalg.norm(action)
    reward_motion_control = 1.0 - min(1.0, action_magnitude / 2.0)  # Normalize action magnitude between 0 and 1
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp_success * reward_grasp_success +
        weight_motion_control * reward_motion_control
    )
    
    # Optional: Additional reward components
    # 1. Bonus for achieving the target qpos
    if self.link_qpos >= self.target_qpos:
        reward += 1.0  # Large bonus for task completion
    
    # 2. Penalty for high drawer velocity to avoid jerky movements
    drawer_velocity = np.linalg.norm(self.link_qvel)
    if drawer_velocity > 0.5:
        reward -= 0.1  # Small penalty for high velocity
    
    return reward