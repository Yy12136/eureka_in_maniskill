import numpy as np
from scipy.spatial.distance import cdist

class BaseEnv(gym.Env):
    def compute_dense_reward(self, action) -> float:
        # Initialize reward
        reward = 0.0

        # Stage 1: Move the gripper close to cube A
        # Calculate the distance between the gripper and cube A
        gripper_pos = self.robot.ee_pose.p  # Gripper position
        cubeA_pos = self.cubeA.pose.p  # Cube A position
        distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)

        # Reward for reducing the distance to cube A
        stage1_reward = -distance_to_cubeA  # Negative distance as reward
        reward += stage1_reward * 0.5  # Weight for stage 1

        # Stage 2: Grasp cube A
        if distance_to_cubeA < 0.05:  # If gripper is close enough to cube A
            is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)  # Check if cube A is grasped
            if is_grasped:
                stage2_reward = 1.0  # Reward for successful grasp
                reward += stage2_reward * 0.3  # Weight for stage 2

                # Stage 3: Lift cube A to the target height
                cubeA_height = cubeA_pos[2]  # Current height of cube A
                target_height = self.cubeA.pose.p[2] + self.goal_height  # Target height
                height_diff = target_height - cubeA_height  # Difference between current and target height

                # Reward for lifting cube A to the target height
                stage3_reward = -np.abs(height_diff)  # Negative height difference as reward
                reward += stage3_reward * 0.2  # Weight for stage 3

        # Regularization of the robot's action
        action_penalty = -np.linalg.norm(action)  # Penalize large actions
        reward += action_penalty * 0.1  # Weight for action regularization

        return reward