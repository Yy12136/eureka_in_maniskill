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
        reward += 1.0  # Sparse reward for task completion
    
    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward