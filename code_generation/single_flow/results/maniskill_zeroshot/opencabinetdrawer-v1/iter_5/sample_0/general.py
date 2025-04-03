import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    ee_coords = self.agent.robot.get_ee_coords()
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
    gripper_openness = self.agent.robot.gripper_openness
    
    # Stage 1: Approach the Handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    if dist_to_handle < 0.05:  # Milestone: Robot is close to the handle
        reward += 1.0  # Sparse reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    if gripper_openness < 0.1:  # Milestone: Gripper is sufficiently closed
        reward += 1.0  # Sparse reward for grasping the handle
    
    # Stage 3: Pull the Drawer
    if drawer_qpos > 0.1 * target_qpos:  # Milestone: Drawer is partially opened
        reward += 1.0  # Sparse reward for starting to pull the drawer
    
    if drawer_qpos > 0.5 * target_qpos:  # Milestone: Drawer is halfway opened
        reward += 2.0  # Sparse reward for significant progress
    
    # Stage 4: Task Completion
    if drawer_qpos >= target_qpos:  # Milestone: Task is completed
        reward += 10.0  # Large sparse reward for task completion
    
    # Penalties
    # Penalize if the drawer is moving in the wrong direction
    if drawer_qpos < 0 and np.abs(drawer_qpos) > 0.01:
        reward -= 0.5  # Penalize for pushing the drawer instead of pulling
    
    # Penalize if the gripper is too open when close to the handle
    if dist_to_handle < 0.1 and gripper_openness > 0.5:
        reward -= 0.5  # Penalize for not closing the gripper when near the handle
    
    return reward