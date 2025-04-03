import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Robot reaches the cabinet
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_position = self.agent.base_pose.p[:2]
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
    if base_to_cabinet_dist < 0.1:  # Threshold for reaching the cabinet
        reward += 0.2  # Reward for reaching the cabinet

    # Milestone 2: Robot grasps the handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    if ee_to_handle_dist < 0.05:  # Threshold for grasping the handle
        reward += 0.3  # Reward for grasping the handle

    # Milestone 3: Robot starts opening the door
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    if door_progress > 0:  # Threshold for starting to open the door
        reward += 0.2  # Reward for starting to open the door

    # Milestone 4: Task completion
    if cabinet_qpos >= target_qpos:  # Threshold for completing the task
        reward += 1.0  # Large reward for task completion

    return reward