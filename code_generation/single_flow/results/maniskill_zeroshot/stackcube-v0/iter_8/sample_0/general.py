import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Grasp Cube A
    if not self.agent.check_grasp(self.cubeA, max_angle=30):
        # Reward for approaching Cube A (optional, can be removed for sparsity)
        gripper_pos = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        dist_gripper_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
        reward += 0.1 / (1.0 + dist_gripper_to_cubeA)
    else:
        # Reward for successfully grasping Cube A
        reward += 1.0

        # Milestone 2: Lift Cube A above Cube B
        cubeB_pos = self.cubeB.pose.p
        height_diff = self.cubeA.pose.p[2] - cubeB_pos[2]
        if height_diff > 0.02 * 2:
            # Reward for lifting Cube A above Cube B
            reward += 1.0

            # Milestone 3: Align Cube A with Cube B
            horizontal_dist = np.linalg.norm(self.cubeA.pose.p[:2] - cubeB_pos[:2])
            if horizontal_dist < 0.02:
                # Reward for aligning Cube A with Cube B
                reward += 1.0

                # Milestone 4: Place Cube A on Cube B
                placement_height_diff = self.cubeA.pose.p[2] - cubeB_pos[2]
                if np.abs(placement_height_diff - 0.02) < 0.01:
                    # Reward for placing Cube A on Cube B
                    reward += 1.0

                    # Milestone 5: Stabilize Cube A
                    if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                        # Reward for successfully placing and stabilizing Cube A
                        reward += 1.0

    # Regularization of the robot's action to encourage smooth movements (optional)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward