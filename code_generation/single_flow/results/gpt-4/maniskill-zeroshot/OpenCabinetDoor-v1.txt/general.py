import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Stage 1: Navigate to the cabinet
    # Reward for reducing the distance between the mobile base and the cabinet
    base_to_cabinet_distance = np.linalg.norm(self.robot.base_position - self.cabinet.pose.p[:2])
    reward += -0.5 * base_to_cabinet_distance  # Weight: 0.5
    
    # Stage 2: Grasp the handle
    # Reward for reducing the distance between the gripper and the cabinet handle
    ee_coords = self.robot.get_ee_coords()
    handle_pcd = self.cabinet.handle.get_world_pcd()
    gripper_to_handle_distance = cdist(ee_coords, handle_pcd).min(axis=1).mean()
    reward += -0.3 * gripper_to_handle_distance  # Weight: 0.3
    
    # Stage 3: Open the door
    # Reward for increasing the qpos of the cabinet door towards the target qpos
    current_qpos = self.cabinet.handle.qpos
    target_qpos = self.cabinet.handle.target_qpos
    door_opening_reward = max(0, current_qpos - target_qpos)
    reward += 0.2 * door_opening_reward  # Weight: 0.2
    
    # Regularization: Penalize large actions
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty  # Weight: 0.1
    
    return reward