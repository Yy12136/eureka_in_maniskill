import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if not self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is not grasped
        gripper_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
        if gripper_to_cubeA_dist < 0.05:  # If gripper is close to Cube A
            if self.robot.check_grasp(self.cubeA, max_angle=30):  # Check if Cube A is grasped
                reward += 1.0  # Reward for successful grasp
        return reward  # Return early if Cube A is not grasped

    # Step 2: Place Cube A on Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is grasped
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is close to Cube B
            if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is stable and released
                reward += 2.0  # Reward for stable placement and release

                # Step 3: Additional reward for precise alignment
                cubeA_center = self.cubeA.pose.p
                cubeB_center = self.cubeB.pose.p
                horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
                if horizontal_dist < 0.01:  # If Cube A is centered on Cube B
                    reward += 0.5  # Reward for precise alignment

                # Step 4: Reward for keeping Cube A upright
                cubeA_rotation = self.cubeA.pose.to_transformation_matrix()[:3, :3]
                desired_rotation = np.eye(3)  # Desired rotation is upright
                rotation_diff = np.linalg.norm(cubeA_rotation - desired_rotation)
                reward += -0.1 * rotation_diff  # Penalize deviation from upright orientation

                # Step 5: Reward for minimizing the time taken to complete the task
                reward += 0.1 * (1 - self.current_step / self.max_steps)  # Reward for completing the task quickly

    return reward