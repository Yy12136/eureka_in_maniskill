import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5) and use "weight_" prefix for all weight variables.
    weight_task_progress = 0.5    # Primary task weight: progress towards opening the drawer
    weight_handle_reach = 0.3     # Secondary task weight: reaching the drawer handle
    weight_stability = 0.2        # Additional weight: maintaining stability during the task
    
    # Initialize reward components (total number <= 5)
    reward_task_progress = 0.0    # Main reward component: progress towards opening the drawer
    reward_handle_reach = 0.0     # Main reward component: reaching the drawer handle
    reward_stability = 0.0        # Main reward component: maintaining stability during the task
    
    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    reward_task_progress = max(0.0, min(1.0, (self.link_qpos - self.target_qpos) / self.target_qpos))
    
    # 2. Handle Reach: Reward based on the distance between the end-effector and the drawer handle
    handle_pos = self.target_link.pose.p
    ee_pos = self.agent.hand.pose.p
    distance_to_handle = np.linalg.norm(handle_pos - ee_pos)
    reward_handle_reach = max(0.0, 1.0 - distance_to_handle / 0.1)  # Normalize distance to 0.1m
    
    # 3. Stability: Penalize large base velocities to maintain stability
    base_velocity = np.linalg.norm(self.agent.base_link.velocity[:2])
    reward_stability = max(0.0, 1.0 - base_velocity / 0.5)  # Normalize velocity to 0.5m/s
    
    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_handle_reach * reward_handle_reach +
        weight_stability * reward_stability
    )
    
    # Optional: Additional reward components
    # 1. Bonus for fully opening the drawer
    if self.link_qpos >= self.target_qpos:
        reward += 1.0
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward