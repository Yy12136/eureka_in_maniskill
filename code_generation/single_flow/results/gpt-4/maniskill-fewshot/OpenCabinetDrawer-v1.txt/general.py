import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Approach the cabinet handle
    # Get the end-effector coordinates
    ee_coords = self.robot.get_ee_coords()
    # Get the point cloud of the cabinet handle in the world frame
    handle_pcd = self.cabinet.handle.get_world_pcd()
    # Calculate the minimum distance between the end-effector and the handle
    dist_to_handle = cdist(ee_coords, handle_pcd).min()
    # Reward for reducing the distance to the handle
    reward += -0.5 * dist_to_handle  # Weight of 0.5 to encourage getting closer

    # Stage 2: Grasp the handle
    # Check if the end-effector is close enough to the handle
    if dist_to_handle < 0.05:  # Threshold for grasping
        # Reward for successfully grasping the handle
        reward += 1.0  # Fixed reward for grasping

        # Stage 3: Pull the drawer
        # Get the current qpos of the cabinet drawer
        current_qpos = self.cabinet.handle.qpos
        # Get the target qpos of the cabinet drawer
        target_qpos = self.cabinet.handle.target_qpos
        # Calculate the difference between current qpos and target qpos
        qpos_diff = target_qpos - current_qpos
        # Reward for pulling the drawer closer to the target qpos
        reward += 0.5 * qpos_diff  # Weight of 0.5 to encourage pulling

        # Check if the task is completed
        if current_qpos >= target_qpos:
            # Large reward for completing the task
            reward += 10.0  # Fixed reward for task completion

    # Regularization of actions
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty  # Weight of 0.01 to penalize large actions

    return reward