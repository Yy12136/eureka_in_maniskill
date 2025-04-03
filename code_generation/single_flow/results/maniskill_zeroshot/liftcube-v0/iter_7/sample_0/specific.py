import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A (ignore z-axis for approach)
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for reducing the distance to Cube A
    approach_reward = -gripper_to_cubeA_distance
    reward += approach_reward * 0.3  # Weight for approach stage

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05:  # Threshold for proximity to grasp
        # Check if the gripper is closed enough to grasp Cube A
        if self.robot.gripper_openness < 0.1:  # Fully closed gripper
            # Check if the grasp is successful
            if self.agent.check_grasp(self.obj, max_angle=30):
                grasp_reward = 1.0  # Large reward for successful grasp
                reward += grasp_reward * 0.4  # Weight for grasp stage
            else:
                # Penalize failed grasp attempts
                grasp_reward = -0.5
                reward += grasp_reward * 0.4
        else:
            # Penalize not closing the gripper
            grasp_reward = -0.2
            reward += grasp_reward * 0.4

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A closer to the goal height
        lift_reward = -height_difference
        reward += lift_reward * 0.3  # Weight for lift stage

        # Step 4: Stabilize Cube A
        if height_difference < 0.01:  # Threshold for reaching the goal height
            # Check if Cube A is static (stable)
            if check_actor_static(self.obj):
                stabilize_reward = 1.0  # Large reward for stabilization
                reward += stabilize_reward * 0.2  # Weight for stabilization stage
            else:
                # Penalize instability
                stabilize_reward = -0.5
                reward += stabilize_reward * 0.2
    else:
        # Penalize dropping Cube A during lifting
        drop_penalty = -1.0
        reward += drop_penalty * 0.3  # Weight for drop penalty

    # Step 5: Regularization of Robot Action
    # Penalize large actions to encourage smooth movements
    action_penalty = -np.linalg.norm(action)
    reward += action_penalty * 0.1  # Weight for action regularization

    return reward