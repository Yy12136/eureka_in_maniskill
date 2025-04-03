import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get the end-effector pose and the cabinet handle pose
    ee_pose = self.robot.ee_pose
    handle_pose = self.cabinet.handle.pose

    # Calculate the distance between the end-effector and the cabinet handle
    ee_coords = self.robot.get_ee_coords()
    handle_pcd = self.cabinet.handle.get_world_pcd()
    dist_ee_handle = cdist(ee_coords, handle_pcd).min(-1).mean()

    # Reward for minimizing the distance between the end-effector and the handle
    reward -= 0.5 * dist_ee_handle  # Weight of 0.5

    # Calculate the difference between the current qpos and the target qpos of the door
    current_qpos = self.cabinet.handle.qpos
    target_qpos = self.cabinet.handle.target_qpos
    qpos_diff = max(0, target_qpos - current_qpos)  # Ensure non-negative

    # Reward for minimizing the qpos difference
    reward -= 0.3 * qpos_diff  # Weight of 0.3

    # Regularization of the robot's action to avoid erratic movements
    action_penalty = np.linalg.norm(action)
    reward -= 0.2 * action_penalty  # Weight of 0.2

    # Staged reward: Encourage the robot to first approach the handle, then open the door
    if dist_ee_handle < 0.05:  # If the end-effector is close to the handle
        reward += 0.1  # Additional reward for being close to the handle

    if qpos_diff < 0.1:  # If the door is almost open
        reward += 0.2  # Additional reward for almost opening the door

    return reward