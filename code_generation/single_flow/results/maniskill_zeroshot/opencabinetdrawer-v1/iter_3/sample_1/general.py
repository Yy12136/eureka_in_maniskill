import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
    
    # Stage 1: Approach the Cabinet
    # Check if the end-effector is close to the handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    if dist_to_handle < 0.05:  # Threshold for being close to the handle
        reward += 1.0  # Milestone reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    # Check if the gripper is sufficiently closed (assuming gripper_openness is normalized)
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    if gripper_openness < 0.1:  # Threshold for grasping
        reward += 1.0  # Milestone reward for grasping the handle
    
    # Stage 3: Pull the Drawer
    # Check if the drawer is partially opened
    if drawer_qpos > 0.1 * target_qpos:  # Threshold for starting to pull
        reward += 1.0  # Milestone reward for starting to pull the drawer
    
    # Stage 4: Task Completion
    # Check if the drawer is fully opened
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Regularization: Penalize large actions (optional, to encourage smoother movements)
    action_penalty = -0.01 * np.linalg.norm(action)  # Small penalty to avoid erratic actions
    reward += action_penalty
    
    return reward