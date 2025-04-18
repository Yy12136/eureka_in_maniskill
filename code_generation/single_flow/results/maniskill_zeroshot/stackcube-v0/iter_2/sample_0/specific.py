import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.cubeA.pose.p
    dist_gripper_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)

    # Reward for reducing the distance to Cube A
    reward += 1.0 / (1.0 + dist_gripper_to_cubeA)

    # Milestone 2: Grasp Cube A
    if dist_gripper_to_cubeA < 0.05:  # If the gripper is close enough to Cube A
        # Check if Cube A is grasped
        is_grasped = self.agent.check_grasp(self.cubeA, max_angle=30)
        if is_grasped:
            # Reward for successfully grasping Cube A
            reward += 1.0

            # Milestone 3: Lift Cube A
            # Calculate the height difference between Cube A and Cube B
            cubeB_pos = self.cubeB.pose.p
            height_diff = cubeA_pos[2] - cubeB_pos[2]
            if height_diff > 0.04:  # If Cube A is lifted above Cube B
                # Reward for lifting Cube A
                reward += 1.0

                # Milestone 4: Place Cube A on Cube B
                # Calculate the horizontal distance between Cube A and Cube B
                horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
                if horizontal_dist < 0.02:  # If Cube A is aligned with Cube B
                    # Reward for aligning Cube A with Cube B
                    reward += 1.0

                    # Milestone 5: Stabilize Cube A
                    # Check if Cube A is static and not grasped
                    if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                        # Reward for successfully placing and stabilizing Cube A
                        reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward