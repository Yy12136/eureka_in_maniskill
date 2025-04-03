import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large reward for task completion

    # Milestone 2: Progress toward task completion
    # Reward proportional to the progress made in opening the door
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    reward += 0.5 * door_progress  # Reward door opening progress

    # Milestone 3: Grasping the handle
    # Reward for successfully grasping the handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:  # If gripper is close to handle
        reward += 0.2  # Bonus for grasping the handle

    # Milestone 4: Approaching the cabinet
    # Reward for getting close to the cabinet
    base_position = self.agent.base_pose.p[:2]
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
    if base_to_cabinet_dist < 0.1:
        reward += 0.1  # Bonus for approaching the cabinet

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward