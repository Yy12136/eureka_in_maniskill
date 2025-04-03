import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    
    # Stage 1: Approach the Cabinet
    # No reward for approaching, as we focus on task completion
    
    # Stage 2: Grasp the Handle
    # No reward for grasping, as we focus on task completion
    
    # Stage 3: Pull the Drawer
    # No reward for pulling, as we focus on task completion
    
    # Stage 4: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 100.0  # Large reward for task completion
    
    # Regularization: Penalize large actions
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty
    
    # Additional Constraints: Penalize if the robot base moves too much
    base_movement_penalty = -0.05 * np.linalg.norm(self.agent.robot.base_velocity)
    reward += base_movement_penalty
    
    return reward