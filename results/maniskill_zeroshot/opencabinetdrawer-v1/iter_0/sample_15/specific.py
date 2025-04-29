import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_task_progress = 0.6  # Primary weight for task progress
    weight_distance_ee = 0.2    # Secondary weight for end-effector distance to handle
    weight_grasp_success = 0.2  # Additional weight for successful grasp

    # Initialize reward components
    reward_task_progress = 0.0  # Main reward component for task progress
    reward_distance_ee = 0.0    # Main reward component for end-effector distance
    reward_grasp_success = 0.0  # Main reward component for grasp success

    # Calculate reward components
    # 1. Task Progress: Reward based on how close the drawer is to the target position
    reward_task_progress = max(0.0, (self.link_qpos - self.target_qpos) / self.target_qpos)

    # 2. End-effector Distance: Reward based on how close the end-effector is to the handle
    handle_pos = self.target_link.pose.p
    ee_pos = self.agent.hand.pose.p
    distance = np.linalg.norm(handle_pos - ee_pos)
    reward_distance_ee = 1.0 / (1.0 + distance)  # Inverse distance reward

    # 3. Grasp Success: Reward if the end-effector is close to the handle and the gripper is closed
    gripper_openness = self.agent.robot.get_qpos()[-1]
    if distance < 0.05 and gripper_openness < 0.1:  # Thresholds for grasp success
        reward_grasp_success = 1.0

    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_distance_ee * reward_distance_ee +
        weight_grasp_success * reward_grasp_success
    )

    # Optional: Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward