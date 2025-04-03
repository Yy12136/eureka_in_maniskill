import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    
    # Stage 1: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 1.0  # Binary reward for task completion
    
    return reward