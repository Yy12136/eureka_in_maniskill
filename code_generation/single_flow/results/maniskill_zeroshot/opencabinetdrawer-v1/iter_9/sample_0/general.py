import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    ee_pose = self.agent.hand.pose
    handle_pose = self.target_link.pose
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
    
    # Stage 1: Approach the Handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    if dist_to_handle < 0.05:  # Milestone: Robot is close to the handle
        reward += 1.0  # Reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    if dist_to_handle < 0.05 and gripper_openness < 0.1:  # Milestone: Gripper is closed and near the handle
        reward += 2.0  # Reward for grasping the handle
    
    # Stage 3: Open the Drawer
    if drawer_qpos >= target_qpos:  # Milestone: Drawer is fully opened
        reward += 10.0  # Large reward for task completion
    
    # Penalize pushing the drawer instead of pulling
    if drawer_qpos < 0 and np.abs(drawer_qpos) > 0.01:
        reward -= 1.0  # Penalize for incorrect direction
    
    return reward