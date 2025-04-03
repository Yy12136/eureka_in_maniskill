import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Task completion
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    # Check if the task is completed
    if current_qpos > target_qpos:
        # Large reward for task completion
        reward += 1.0

    # Stage 2: Minimal guidance during the process
    # Reward for reducing the distance to the handle (optional, for initial guidance)
    ee_pcd = self.robot.get_world_pcd()  # Get the gripper's point cloud in the world frame
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)  # Get the handle's point cloud in the world frame
    min_distance = cdist(ee_pcd, handle_pcd).min()  # Calculate the minimum distance

    # Small reward for approaching the handle (optional)
    reward += 0.1 * (1 - np.tanh(10 * min_distance))

    # Regularization of the robot's action to encourage smooth movements (optional)
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward