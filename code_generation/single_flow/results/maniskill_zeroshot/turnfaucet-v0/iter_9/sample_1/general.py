import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Task completion
    current_qpos = self.current_angle
    target_qpos = self.target_angle

    # Large reward for task completion
    if current_qpos > target_qpos:
        reward += 1.0  # Sparse reward for success
        return reward  # Return immediately after task completion

    # Stage 2: Grasp the handle (intermediate reward)
    # Calculate the minimum distance between the gripper and the handle's point cloud
    ee_pcd = self.robot.get_world_pcd()  # Get the gripper's point cloud in the world frame
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)  # Get the handle's point cloud in the world frame
    min_distance = cdist(ee_pcd, handle_pcd).min()  # Calculate the minimum distance

    if min_distance < 0.1:  # If the end-effector is close enough to the handle
        # Check if the handle is grasped
        is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
        if is_grasped:
            reward += 0.1  # Small intermediate reward for grasping

    # Stage 3: Rotate the handle (intermediate reward)
    if is_grasped:
        # Normalize rotation progress between 0 and 1
        rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)
        reward += 0.1 * rotation_progress  # Small intermediate reward for rotation progress

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward