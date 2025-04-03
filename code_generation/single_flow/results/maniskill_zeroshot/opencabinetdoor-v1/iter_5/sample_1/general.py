import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Robot base is close to the cabinet
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_position = self.agent.base_pose.p[:2]
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
    if base_to_cabinet_dist < 0.1:  # Threshold for being close to the cabinet
        reward += 0.1  # Sparse reward for approaching the cabinet

    # Milestone 2: End-effector is close to the cabinet handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = np.min(np.linalg.norm(ee_coords - handle_pcd, axis=1))
    if ee_to_handle_dist < 0.05:  # Threshold for being close to the handle
        reward += 0.2  # Sparse reward for grasping the handle

    # Milestone 3: Door is opened beyond the target qpos
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large sparse reward for task completion

    return reward