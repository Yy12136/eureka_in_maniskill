import numpy as np

def compute_sparse_reward(self) -> float:
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    
    # Initialize reward
    reward = 0.0
    
    # Task Completion
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    return reward