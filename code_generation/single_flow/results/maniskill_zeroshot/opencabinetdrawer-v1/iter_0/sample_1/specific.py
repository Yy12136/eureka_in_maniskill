import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
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
    
    # Stage 1: Approach the Cabinet
    # Reward for reducing the distance between the end-effector and the handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    reward += -0.5 * dist_to_handle  # Encourage closer approach
    
    # Stage 2: Grasp the Handle
    # Reward for aligning the gripper with the handle
    alignment_reward = -np.linalg.norm(ee_pose.p - handle_pose.p)
    reward += 0.3 * alignment_reward
    
    # Reward for closing the gripper
    gripper_reward = -gripper_openness  # Encourage closing the gripper
    reward += 0.2 * gripper_reward
    
    # Stage 3: Pull the Drawer
    # Reward for increasing the drawer's qpos
    drawer_movement_reward = drawer_qpos
    reward += 0.5 * drawer_movement_reward
    
    # Stage 4: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Regularization: Penalize large actions
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward