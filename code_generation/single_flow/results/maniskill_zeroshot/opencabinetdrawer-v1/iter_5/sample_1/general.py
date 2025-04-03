import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    ee_pose = self.agent.hand.pose
    handle_pose = self.target_link.pose
    gripper_openness = self.agent.robot.gripper_openness
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    ee_coords = self.agent.robot.get_ee_coords()
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)
    
    # Stage 1: Approach the Cabinet
    # Reward only when the end-effector is close to the handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    if dist_to_handle < 0.05:  # If the robot is close to the handle
        reward += 1.0  # Milestone reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    # Reward only when the gripper is sufficiently closed
    if gripper_openness < 0.1:  # If the gripper is sufficiently closed
        reward += 1.0  # Milestone reward for grasping the handle
    
    # Stage 3: Pull the Drawer
    # Reward for significant progress in pulling the drawer
    if drawer_qpos > 0.5 * target_qpos:  # If the drawer is halfway opened
        reward += 2.0  # Milestone reward for significant progress
    
    # Stage 4: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Regularization: Penalize large actions (optional, to encourage smooth control)
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward