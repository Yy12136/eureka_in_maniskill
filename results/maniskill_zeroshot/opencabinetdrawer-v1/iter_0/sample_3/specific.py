import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_task_progress = 0.6  # Primary weight for task progress
    reward_handle_reach = 0.3  # Secondary weight for reaching the handle
    reward_control = 0.1  # Additional weight for motion control

    # Initialize reward components
    reward_task_progress = 0.0
    reward_handle_reach = 0.0
    reward_control = 0.0

    # Calculate task progress reward
    current_qpos = self.link_qpos
    target_qpos = self.target_qpos
    progress = max(0, (current_qpos - target_qpos) / target_qpos)  # Normalized progress
    reward_task_progress = progress

    # Calculate handle reach reward
    handle_pos = self.target_link.pose.p
    ee_pos = self.agent.hand.pose.p
    distance_to_handle = np.linalg.norm(handle_pos - ee_pos)
    reward_handle_reach = 1.0 / (1.0 + distance_to_handle)  # Inverse distance reward

    # Calculate motion control reward
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    velocity_magnitude = np.linalg.norm(joint_velocities)
    reward_control = 1.0 / (1.0 + velocity_magnitude)  # Inverse velocity reward

    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        reward_handle_reach * reward_handle_reach +
        reward_control * reward_control
    )

    # Optional: Penalty for large actions (regularization)
    action_magnitude = np.linalg.norm(action)
    penalty_action = 0.01 * action_magnitude
    reward -= penalty_action

    return reward