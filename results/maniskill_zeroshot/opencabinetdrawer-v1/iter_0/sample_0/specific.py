import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_task_progress = 0.6000    # Primary weight for task progress
    weight_handle_reach = 0.3000    # Secondary weight for reaching the handle
    weight_control = 0.1000    # Additional weight for motion control

    # Initialize reward components
    reward_task_progress = 0.0  # Main reward for task progress
    reward_handle_reach = 0.0   # Main reward for reaching the handle
    reward_control = 0.0        # Main reward for motion control

    # Calculate reward components
    # Task progress: based on how close the drawer is to the target position
    reward_task_progress = max(0.0, (self.link_qpos - self.target_qpos) / self.target_qpos)

    # Handle reach: based on the distance between the end-effector and the handle
    ee_pos = self.agent.hand.pose.p
    handle_pos = self.target_link.pose.p
    distance = np.linalg.norm(ee_pos - handle_pos)
    reward_handle_reach = 1.0 / (1.0 + distance)

    # Motion control: penalize large actions to encourage smooth motion
    action_magnitude = np.linalg.norm(action)
    reward_control = 1.0 / (1.0 + action_magnitude)

    # Combine main rewards
    reward = (
        weight_task_progress * reward_task_progress +
        weight_handle_reach * reward_handle_reach +
        weight_control * reward_control
    )

    # Optional: Additional reward components
    # Bonus for maintaining the drawer above a certain height (if applicable)
    # Penalty for large base velocity to encourage stable movement
    base_velocity = np.linalg.norm(self.agent.base_link.velocity[:2])
    if base_velocity > 0.1:
        reward -= 0.05 * base_velocity

    return reward