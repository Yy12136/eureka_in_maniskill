import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # Get the current qpos of the faucet handle
    current_qpos = self.faucet.handle.qpos
    target_qpos = self.faucet.handle.target_qpos

    # Check if the task is completed
    if current_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
        return reward

    # Stage 1: Approach the faucet handle
    # Calculate the distance between the gripper and the faucet handle
    gripper_pos = self.robot.ee_pose.p
    handle_pcd = self.faucet.handle.get_world_pcd()
    min_dist = np.min(np.linalg.norm(handle_pcd - gripper_pos, axis=1))

    # Reward for being close to the handle
    if min_dist < 0.1:  # Threshold for being "close"
        reward += 1.0

    # Stage 2: Grasp the faucet handle
    is_grasped = self.robot.check_grasp(self.faucet.handle, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # Stage 3: Rotate the faucet handle
    if is_grasped:
        # Reward proportional to the progress in rotating the handle
        rotation_progress = current_qpos / target_qpos
        reward += 2.0 * rotation_progress  # Linear reward for progress

    return reward