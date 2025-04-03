import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05:  # Threshold for proximity to grasp
        if self.robot.gripper_openness < 0.1:  # Fully closed gripper
            if self.agent.check_grasp(self.obj, max_angle=30):
                # Large reward for successful grasp
                reward += 1.0
            else:
                # Penalize failed grasp attempts
                reward -= 0.5
        else:
            # Penalize not closing the gripper
            reward -= 0.2

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        if height_difference < 0.01:  # Threshold for reaching the goal height
            # Large reward for successful lift and stabilization
            reward += 1.0
        else:
            # Small reward for progress towards the goal height
            reward += 0.1 * (1 - height_difference / self.goal_height)
    else:
        # Penalize dropping Cube A during lifting
        reward -= 1.0

    return reward