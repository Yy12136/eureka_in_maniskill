import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Approach the handle
    # Calculate the distance between the gripper and the handle's point cloud
    ee_pose = self.robot.ee_pose.p
    handle_pcd = self.faucet.handle.get_world_pcd()
    dist_to_handle = cdist(ee_pose[np.newaxis, :], handle_pcd).min()

    # Reward for reducing the distance to the handle
    reward += -0.1 * dist_to_handle  # Weight can be adjusted

    # Stage 2: Grasp the handle
    if dist_to_handle < 0.05:  # Threshold for being close enough to grasp
        is_grasped = self.robot.check_grasp(self.faucet.handle, max_angle=30)
        if is_grasped:
            reward += 1.0  # Reward for successful grasp

            # Stage 3: Rotate the handle
            current_qpos = self.faucet.handle.qpos
            target_qpos = self.faucet.handle.target_qpos
            qpos_diff = target_qpos - current_qpos

            # Reward for increasing the qpos toward the target
            reward += 0.5 * (current_qpos / target_qpos)  # Weight can be adjusted

            # Stage 4: Task completion
            if current_qpos >= target_qpos:
                reward += 10.0  # Large reward for task completion

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward