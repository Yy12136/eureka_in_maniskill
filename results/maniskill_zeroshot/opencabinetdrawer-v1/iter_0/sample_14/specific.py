import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5) and use "weight_" prefix for all weight variables.
    weight_task_progress = 0.5    # Primary weight for task progress
    weight_handle_reach = 0.3     # Secondary weight for reaching the handle
    weight_grasp_success = 0.2    # Additional weight for successful grasp
    # Note: weight_task_progress + weight_handle_reach + weight_grasp_success = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_handle_reach = 0.0     # Main reward component for reaching the handle
    reward_grasp_success = 0.0    # Main reward component for successful grasp
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    reward_task_progress = max(0, (self.link_qpos - self.target_qpos) / self.target_qpos)
    
    # 2. Handle Reach: Reward based on the distance between the end-effector and the handle
    ee_pos = self.agent.hand.pose.p
    handle_pos = self.target_link.pose.p
    distance_to_handle = np.linalg.norm(ee_pos - handle_pos)
    reward_handle_reach = 1.0 / (1.0 + distance_to_handle)
    
    # 3. Grasp Success: Reward based on the gripper's grasp state
    gripper_openness = self.agent.robot.get_qpos()[-1]
    reward_grasp_success = 1.0 if gripper_openness < 0.01 else 0.0
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_handle_reach * reward_handle_reach +
        weight_grasp_success * reward_grasp_success
    )
    
    # Optional: Additional reward components
    # 1. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward