import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the end-effector and cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    distance_to_cubeA = np.linalg.norm(ee_pose_wrt_cubeA.p)
    approach_reward = -distance_to_cubeA  # Closer is better
    reward += approach_reward * 0.2  # Weight for approach stage

    # Step 2: Grasp Cube A
    if distance_to_cubeA < 0.05:  # If the end-effector is close enough to cube A
        grasp_success = self.agent.check_grasp(self.obj, max_angle=30)
        if grasp_success:
            grasp_reward = 1.0  # Reward for successful grasp
            reward += grasp_reward * 0.3  # Weight for grasp stage

            # Step 3: Lift Cube A
            cubeA_height = self.obj.pose.p[2]  # Z-coordinate of cube A
            if cubeA_height > 0.02:  # If cube A is lifted
                lift_reward = 1.0  # Reward for lifting
                reward += lift_reward * 0.2  # Weight for lift stage

                # Step 4: Move Cube A to Goal
                distance_to_goal = np.linalg.norm(self.obj.pose.p - self.goal_pos)
                move_reward = -distance_to_goal  # Closer to goal is better
                reward += move_reward * 0.2  # Weight for move stage

                # Step 5: Place Cube A at Goal
                if distance_to_goal < 0.02:  # If cube A is close to the goal position
                    place_reward = 1.0  # Reward for placing
                    reward += place_reward * 0.1  # Weight for place stage

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -np.linalg.norm(action) * 0.01
    reward += action_penalty

    return reward