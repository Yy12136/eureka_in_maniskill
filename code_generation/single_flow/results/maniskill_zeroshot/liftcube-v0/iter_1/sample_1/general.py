import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for reaching the proximity of Cube A
    if gripper_to_cubeA_distance < 0.05:  # Threshold for proximity to grasp
        reward += 0.3  # Sparse reward for approaching Cube A

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05:  # Threshold for proximity to grasp
        if self.robot.gripper_openness < 0.1:  # Fully closed gripper
            if self.agent.check_grasp(self.obj, max_angle=30):  # Successful grasp
                reward += 0.4  # Sparse reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)
        if height_difference < 0.01:  # Threshold for reaching the goal height
            reward += 0.3  # Sparse reward for lifting Cube A to the goal height

    # Step 4: Stabilize Cube A
    if height_difference < 0.01:  # Threshold for reaching the goal height
        if check_actor_static(self.obj):  # Check if Cube A is static (stable)
            reward += 0.2  # Sparse reward for stabilizing Cube A

    # Step 5: Regularization of Robot Action (Optional)
    # Penalize large actions to encourage smooth movements (can be omitted for sparse reward)
    action_penalty = -np.linalg.norm(action)
    reward += action_penalty * 0.1  # Weight for action regularization

    return reward