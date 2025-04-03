import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    
    # Stage 1: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 100.0  # Large reward for task completion
    
    # Stage 2: Milestone Rewards
    # Encourage progress toward the goal
    if drawer_qpos > 0.5 * target_qpos:  # If the drawer is halfway opened
        reward += 10.0  # Milestone reward for significant progress
    
    # Stage 3: Initial Progress
    # Encourage starting the task
    if drawer_qpos > 0.1 * target_qpos:  # If the drawer is slightly opened
        reward += 5.0  # Milestone reward for initial progress
    
    # Regularization: Penalize large actions (optional)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward