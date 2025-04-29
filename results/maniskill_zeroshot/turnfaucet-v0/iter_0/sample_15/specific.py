import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_progress = 0.6  # Primary weight for task progress
    weight_grasp = 0.3     # Secondary weight for grasp state
    weight_control = 0.1   # Additional weight for motion control

    # Initialize reward components
    reward_progress = 0.0  # Task progress reward
    reward_grasp = 0.0     # Grasp state reward
    reward_control = 0.0   # Motion control reward

    # Calculate task progress reward
    progress = (self.current_angle - self.target_angle) / self.target_angle
    reward_progress = max(0.0, min(1.0, progress))  # Normalize progress between 0 and 1

    # Calculate grasp state reward
    if self.agent.check_grasp(self.target_link):
        reward_grasp = 1.0  # Full reward if the handle is grasped

    # Calculate motion control reward
    action_magnitude = sum([a ** 2 for a in action]) ** 0.5
    reward_control = 1.0 / (1.0 + action_magnitude)  # Reward smoother actions

    # Combine main rewards
    reward = (
        weight_progress * reward_progress +
        weight_grasp * reward_grasp +
        weight_control * reward_control
    )

    # Optional: Additional reward components
    # Bonus for completing the task
    if self.current_angle >= self.target_angle:
        reward += 1.0  # Add a bonus for task completion

    return reward