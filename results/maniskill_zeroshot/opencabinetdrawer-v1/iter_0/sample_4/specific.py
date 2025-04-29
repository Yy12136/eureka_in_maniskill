import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_task_progress = 0.6  # Primary weight for task progress
    weight_grasp_state = 0.2    # Secondary weight for grasp state
    weight_motion_control = 0.2 # Additional weight for motion control

    # Initialize reward components
    reward_task_progress = 0.0  # Main reward component for task progress
    reward_grasp_state = 0.0    # Main reward component for grasp state
    reward_motion_control = 0.0 # Main reward component for motion control

    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    reward_task_progress = max(0, (self.link_qpos - self.target_qpos) / self.target_qpos)

    # 2. Grasp State: Reward for maintaining a stable grasp on the handle
    # Assuming grasp success is indicated by the gripper being close to the handle
    handle_pos = self.target_link.pose.p
    gripper_pos = self.agent.hand.pose.p
    distance_to_handle = np.linalg.norm(handle_pos - gripper_pos)
    reward_grasp_state = 1.0 - min(1.0, distance_to_handle / 0.1)  # Normalize distance to [0, 1]

    # 3. Motion Control: Reward for smooth and controlled motion
    # Penalize large joint velocities to encourage smooth movement
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward_motion_control = 1.0 - min(1.0, np.linalg.norm(joint_velocities) / 5.0)  # Normalize velocity to [0, 1]

    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_grasp_state * reward_grasp_state +
        weight_motion_control * reward_motion_control
    )

    # Optional: Additional reward components
    # 1. Bonus for achieving the target position
    if self.link_qpos >= self.target_qpos:
        reward += 1.0  # Large bonus for task completion

    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward