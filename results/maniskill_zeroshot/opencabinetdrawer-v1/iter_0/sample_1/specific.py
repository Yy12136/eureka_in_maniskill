import numpy as np

def compute_dense_reward(self, action) -> float:
    weight_task_progress = 0.5000    # Primary weight for task progress
    weight_distance_ee = 0.3000    # Secondary weight for end-effector distance to handle
    weight_grasp_success = 0.2000    # Additional weight for successful grasp
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component for task progress
    reward_distance_ee = 0.0      # Main reward component for end-effector distance
    reward_grasp_success = 0.0    # Main reward component for grasp success
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    progress = (self.link_qpos - self.target_qpos) / self.target_qpos
    reward_task_progress = max(0.0, min(1.0, progress))
    
    # 2. End-effector Distance: Reward based on how close the end-effector is to the handle
    ee_pos = self.agent.hand.pose.p
    handle_pos = self.target_link.pose.p
    distance = np.linalg.norm(ee_pos - handle_pos)
    reward_distance_ee = 1.0 / (1.0 + distance)
    
    # 3. Grasp Success: Reward if the end-effector is close to the handle and the gripper is closed
    gripper_openness = self.agent.robot.get_qpos()[-1]
    if distance < 0.05 and gripper_openness < 0.1:  # Assuming 0.1 is the threshold for a closed gripper
        reward_grasp_success = 1.0
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_distance_ee * reward_distance_ee +
        weight_grasp_success * reward_grasp_success
    )
    
    # Optional: Additional reward components
    # 1. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward