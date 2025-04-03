import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    gripper_openness = self.agent.robot.gripper_openness
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    ee_coords = self.agent.robot.get_ee_coords()
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
    
    # Stage 1: Approach the Handle
    # Reward for being close to the handle (sparse milestone)
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    if dist_to_handle < 0.05:  # If the robot is close to the handle
        reward += 1.0  # Milestone reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    # Reward for closing the gripper sufficiently (sparse milestone)
    if gripper_openness < 0.1:  # If the gripper is sufficiently closed
        reward += 2.0  # Milestone reward for grasping the handle
    
    # Stage 3: Pull the Drawer
    # Reward for significant progress in pulling the drawer (sparse milestone)
    if drawer_qpos > 0.5 * target_qpos:  # If the drawer is halfway opened
        reward += 3.0  # Milestone reward for significant progress
    
    # Stage 4: Task Completion
    # Large reward for completing the task (sparse milestone)
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Penalties for undesirable behaviors
    # Penalize if the drawer is moving in the wrong direction
    if drawer_qpos < 0 and np.abs(drawer_qpos) > 0.01:
        reward -= 1.0  # Penalize for pushing the drawer instead of pulling
    
    # Penalize if the robot base is too far from the cabinet
    base_to_cabinet_dist = np.linalg.norm(self.agent.robot.base_position - self.cabinet.pose.p[:2])
    if base_to_cabinet_dist > 0.5:
        reward -= 1.0  # Penalize for being too far from the cabinet
    
    return reward