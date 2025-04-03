import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
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
    # Reward for reducing the distance between the end-effector and the handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    approach_reward = -0.5 * dist_to_handle  # Encourage closer approach
    reward += approach_reward
    
    # Milestone reward for approaching the handle
    if dist_to_handle < 0.05:  # If the robot is close to the handle
        reward += 1.0  # Milestone reward for approaching the handle
    
    # Stage 2: Grasp the Handle
    # Reward for aligning the gripper with the handle
    alignment_reward = -np.linalg.norm(ee_pose.p - handle_pose.p)
    reward += 0.3 * alignment_reward
    
    # Reward for closing the gripper
    gripper_reward = -gripper_openness  # Encourage closing the gripper
    reward += 0.2 * gripper_reward
    
    # Milestone reward for grasping the handle
    if gripper_openness < 0.1:  # If the gripper is sufficiently closed
        reward += 1.0  # Milestone reward for grasping the handle
    
    # Stage 3: Pull the Drawer
    # Reward for increasing the drawer's qpos
    drawer_movement_reward = drawer_qpos
    reward += 0.5 * drawer_movement_reward
    
    # Milestone reward for starting to pull the drawer
    if drawer_qpos > 0.1 * target_qpos:  # If the drawer is partially opened
        reward += 1.0  # Milestone reward for starting to pull the drawer
    
    # Milestone reward for significant progress in pulling the drawer
    if drawer_qpos > 0.5 * target_qpos:  # If the drawer is halfway opened
        reward += 2.0  # Milestone reward for significant progress
    
    # Stage 4: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Regularization: Penalize large actions
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty
    
    # Additional Constraints: Penalize if the robot base moves too much
    base_movement_penalty = -0.05 * np.linalg.norm(self.agent.robot.base_velocity)
    reward += base_movement_penalty
    
    # Penalize if the gripper is too open when close to the handle
    if dist_to_handle < 0.1 and gripper_openness > 0.5:
        reward -= 0.5  # Penalize for not closing the gripper when near the handle
    
    # Penalize if the drawer is moving in the wrong direction
    if drawer_qpos < 0 and np.abs(drawer_qpos) > 0.01:
        reward -= 0.5  # Penalize for pushing the drawer instead of pulling
    
    # Penalize if the robot is not aligned with the handle when pulling
    if drawer_qpos > 0.1 * target_qpos and dist_to_handle > 0.1:
        reward -= 0.5  # Penalize for misalignment during pulling
    
    # Penalize if the robot base is too far from the cabinet
    base_to_cabinet_dist = np.linalg.norm(self.agent.robot.base_position - self.cabinet.pose.p[:2])
    if base_to_cabinet_dist > 0.5:
        reward -= 0.5  # Penalize for being too far from the cabinet
    
    # Penalize if the robot is not pulling the drawer smoothly
    if drawer_qpos > 0.1 * target_qpos and np.abs(self.link_qvel) > 0.1:
        reward -= 0.5  # Penalize for jerky drawer movement
    
    return reward