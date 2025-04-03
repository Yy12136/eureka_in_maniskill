import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Task completion
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    if current_qpos > target_qpos:
        # Large reward for task completion
        reward += 1.0
        return reward  # Return immediately after task completion

    # Stage 2: Grasp the handle (optional, if needed)
    is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
    if is_grasped:
        # Small reward for successful grasp
        reward += 0.1

    # Stage 3: Rotate the handle (optional, if needed)
    if is_grasped:
        rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)
        # Small reward for making progress towards rotation
        reward += 0.1 * rotation_progress

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward