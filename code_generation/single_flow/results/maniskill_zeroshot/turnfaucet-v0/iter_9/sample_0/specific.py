import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize the reward
    reward = 0.0

    # Stage 1: Approach the faucet handle
    # Calculate the minimum distance between the gripper and the handle's point cloud
    ee_pcd = self.robot.get_world_pcd()  # Get the gripper's point cloud in the world frame
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_link_pcd)  # Get the handle's point cloud in the world frame
    min_distance = cdist(ee_pcd, handle_pcd).min()  # Calculate the minimum distance

    # Reward for reducing the distance to the handle
    # Use a tanh function to normalize the reward between 0 and 1
    reward += 0.3 * (1 - np.tanh(10 * min_distance))

    # Stage 2: Align the gripper with the handle
    # Calculate the orientation difference between the gripper and the handle
    ee_quat = self.tcp.pose.q
    handle_quat = self.target_link.pose.q
    quat_similarity = np.abs(np.dot(ee_quat, handle_quat))  # Quaternion dot product to measure orientation similarity
    reward += 0.2 * quat_similarity

    # Stage 3: Grasp the handle
    if min_distance < 0.1:  # If the end-effector is close enough to the handle
        # Check if the handle is grasped
        is_grasped = self.agent.check_grasp(self.target_link, max_angle=30)
        if is_grasped:
            # Reward for successful grasp
            reward += 0.3

            # Stage 4: Rotate the handle
            current_qpos = self.current_angle
            target_qpos = self.target_angle
            # Normalize rotation progress between 0 and 1
            rotation_progress = max(0, (current_qpos - target_qpos) / target_qpos)

            # Reward for rotating the handle towards the target
            reward += 0.2 * rotation_progress

            # Stage 5: Task completion
            if current_qpos > target_qpos:
                # Large reward for task completion
                reward += 1.0

    # Stage 6: Encourage smooth and efficient movements
    # Penalize large actions to encourage smooth movements
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    # Stage 7: Penalize excessive gripper openness
    # Encourage the gripper to be closed when grasping
    if min_distance < 0.1 and not is_grasped:
        gripper_penalty = 0.1 * self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
        reward -= gripper_penalty

    return reward