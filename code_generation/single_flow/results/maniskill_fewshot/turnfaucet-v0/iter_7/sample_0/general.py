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
    approach_reward = 1 - np.tanh(5 * min_dist)
    reward += approach_reward

    # Stage 2: Grasp the faucet handle
    is_grasped = self.robot.check_grasp(self.faucet.handle, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # Stage 3: Rotate the faucet handle
    if is_grasped:
        # Calculate the progress in rotating the handle
        rotation_progress = current_qpos / target_qpos
        rotation_reward = np.tanh(5 * rotation_progress)
        reward += rotation_reward

        # Additional reward for maintaining a stable grasp while rotating
        gripper_openness = self.robot.gripper_openness
        if gripper_openness < 0.1:  # Ensure the gripper is closed
            reward += 0.5 * (1 - gripper_openness)

        # Penalize fast rotation to encourage smooth movements
        handle_velocity = np.abs(self.faucet.handle.qvel)
        velocity_penalty = -0.1 * handle_velocity
        reward += velocity_penalty

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward