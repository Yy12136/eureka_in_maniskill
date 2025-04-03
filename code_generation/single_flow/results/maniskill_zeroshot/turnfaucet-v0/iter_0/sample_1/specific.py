import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Approach the faucet handle
    # Calculate the distance between the end-effector and the faucet handle
    ee_pose = self.tcp.pose.p
    handle_pose = self.target_link.pose.p
    distance_to_handle = np.linalg.norm(ee_pose - handle_pose)

    # Reward for reducing the distance to the handle
    reward += 0.5 * (1 - np.tanh(10 * distance_to_handle))

    # Stage 2: Grasp the handle
    if distance_to_handle < 0.1:  # If the end-effector is close enough to the handle
        # Check if the handle is grasped
        is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
        if is_grasped:
            # Reward for successful grasp
            reward += 0.3

            # Stage 3: Rotate the handle
            current_qpos = self.current_angle
            target_qpos = self.target_angle
            rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)

            # Reward for rotating the handle towards the target
            reward += 0.2 * rotation_progress

            # Stage 4: Task completion
            if current_qpos > target_qpos:
                # Large reward for task completion
                reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward