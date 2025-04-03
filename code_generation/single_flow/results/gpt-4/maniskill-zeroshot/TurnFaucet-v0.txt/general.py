import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the faucet handle
    # Calculate the distance between the gripper and the faucet handle
    gripper_pcd = np.vstack([self.robot.lfinger.pose.p, self.robot.rfinger.pose.p])  # Get gripper finger positions
    handle_pcd = self.faucet.handle.get_world_pcd()  # Get faucet handle point cloud
    dist = cdist(gripper_pcd, handle_pcd).min()  # Minimum distance between gripper and handle

    # Reward for reducing the distance
    approach_reward = -dist  # Negative distance encourages minimizing it
    reward += approach_reward * 0.5  # Weight for approach stage

    # Stage 2: Grasp the faucet handle
    if dist < 0.05:  # If the gripper is close enough to the handle
        grasp_success = self.robot.check_grasp(self.faucet.handle, max_angle=30)  # Check if grasp is successful
        if grasp_success:
            grasp_reward = 1.0  # Reward for successful grasp
            reward += grasp_reward * 0.3  # Weight for grasp stage

            # Stage 3: Rotate the faucet handle toward the target position
            current_qpos = self.faucet.handle.qpos  # Current joint position of the handle
            target_qpos = self.faucet.handle.target_qpos  # Target joint position
            rotation_progress = current_qpos / target_qpos  # Progress toward target

            # Reward for rotating the handle
            rotation_reward = rotation_progress  # Directly proportional to progress
            reward += rotation_reward * 0.2  # Weight for rotation stage

            # Stage 4: Task completion bonus
            if current_qpos >= target_qpos:  # If the task is completed
                completion_bonus = 10.0  # Large bonus for completion
                reward += completion_bonus

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = -np.linalg.norm(action)  # Penalize the magnitude of the action
    reward += action_penalty * 0.1  # Weight for action regularization

    return reward