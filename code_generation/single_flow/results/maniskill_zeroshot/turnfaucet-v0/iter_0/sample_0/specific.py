import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get the faucet handle's point cloud in the world frame
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)

    # Get the robot's end-effector pose and finger positions
    ee_pose = self.tcp.pose
    lfinger_pcd = transform_points(self.lfinger.pose.to_transformation_matrix(), self.lfinger_pcd)
    rfinger_pcd = transform_points(self.rfinger.pose.to_transformation_matrix(), self.rfinger_pcd)

    # Calculate the minimum distance between the end-effector and the handle
    ee_to_handle_dist = cdist(np.array([ee_pose.p]), handle_pcd).min()

    # Calculate the minimum distance between the fingers and the handle
    lfinger_to_handle_dist = cdist(lfinger_pcd, handle_pcd).min()
    rfinger_to_handle_dist = cdist(rfinger_pcd, handle_pcd).min()
    finger_to_handle_dist = min(lfinger_to_handle_dist, rfinger_to_handle_dist)

    # Stage 1: Approach the handle
    if not self.agent.check_grasp(self.target_link):
        # Reward for reducing the distance to the handle
        reward += -ee_to_handle_dist * 0.5
        # Reward for reducing the distance between fingers and handle
        reward += -finger_to_handle_dist * 0.5
    else:
        # Stage 2: Grasp the handle
        # Reward for successfully grasping the handle
        reward += 1.0

        # Stage 3: Rotate the handle
        # Get the current and target qpos of the handle
        current_qpos = self.current_angle
        target_qpos = self.target_angle

        # Reward for rotating the handle closer to the target position
        rotation_progress = (current_qpos - target_qpos) / target_qpos
        reward += -abs(rotation_progress) * 0.5

        # Stage 4: Task completion
        if current_qpos >= target_qpos:
            # Large reward for completing the task
            reward += 10.0

    # Regularization of the robot's action to encourage smooth movements
    reward -= 0.01 * np.linalg.norm(action)

    return reward