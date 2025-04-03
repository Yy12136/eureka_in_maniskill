import numpy as np

def compute_dense_reward(self, action) -> float:
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

    # Milestone: Ensure the gripper is oriented correctly towards the handle
    gripper_pose = self.robot.ee_pose
    handle_pose = self.faucet.handle.pose
    gripper_to_handle = handle_pose.inv() * gripper_pose
    gripper_z = gripper_to_handle.to_transformation_matrix()[:3, 2]
    handle_z = np.array([0, 0, 1])  # Assuming the handle's z-axis is along the world z-axis
    angle = np.arccos(np.clip(np.dot(gripper_z, handle_z), -1.0, 1.0))
    orientation_reward = 1 - np.tanh(5 * angle)
    reward += orientation_reward

    # Stage 2: Grasp the faucet handle
    is_grasped = self.robot.check_grasp(self.faucet.handle, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Milestone: Encourage the gripper to maintain a stable position relative to the handle
        gripper_to_handle_distance = np.linalg.norm(gripper_pos - handle_pose.p)
        stable_position_reward = 1 - np.tanh(5 * gripper_to_handle_distance)
        reward += stable_position_reward

        # Milestone: Encourage the gripper to maintain a stable grasp
        gripper_openness = self.robot.gripper_openness
        if gripper_openness < 0.1:  # Ensure the gripper is closed
            reward += 0.5 * (1 - gripper_openness)

    # Stage 3: Rotate the faucet handle
    if is_grasped:
        # Calculate the progress in rotating the handle
        rotation_progress = current_qpos / target_qpos
        rotation_reward = np.tanh(5 * rotation_progress)
        reward += rotation_reward

        # Penalize fast rotation to encourage smooth movements
        handle_velocity = np.abs(self.faucet.handle.qvel)
        velocity_penalty = -0.1 * handle_velocity
        reward += velocity_penalty

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    # Additional Milestone: Encourage the robot to maintain a stable joint configuration
    neutral_qpos = np.zeros(7)  # Assuming neutral joint configuration is all zeros
    joint_deviation = np.linalg.norm(self.robot.qpos - neutral_qpos)
    joint_stability_reward = -0.1 * joint_deviation
    reward += joint_stability_reward

    # Additional Milestone: Encourage the robot to maintain a stable velocity
    joint_velocity = np.linalg.norm(self.robot.qvel)
    velocity_stability_reward = -0.05 * joint_velocity
    reward += velocity_stability_reward

    # Additional Milestone: Encourage smooth gripper movement
    gripper_delta = np.abs(self.robot.gripper_openness - self.previous_gripper_openness)
    gripper_smoothness_reward = -0.1 * gripper_delta
    reward += gripper_smoothness_reward

    # Update previous gripper openness for the next step
    self.previous_gripper_openness = self.robot.gripper_openness

    # Additional Milestone: Encourage the robot to maintain a stable end-effector position
    neutral_ee_pos = np.array([0.5, 0.0, 0.5])  # Assuming neutral end-effector position
    ee_deviation = np.linalg.norm(gripper_pos - neutral_ee_pos)
    ee_stability_reward = -0.1 * ee_deviation
    reward += ee_stability_reward

    # Additional Milestone: Encourage the robot to maintain a stable orientation of the end-effector
    neutral_ee_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Assuming neutral end-effector orientation
    ee_orientation_deviation = np.linalg.norm(self.robot.ee_pose.q - neutral_ee_orientation)
    ee_orientation_stability_reward = -0.1 * ee_orientation_deviation
    reward += ee_orientation_stability_reward

    return reward