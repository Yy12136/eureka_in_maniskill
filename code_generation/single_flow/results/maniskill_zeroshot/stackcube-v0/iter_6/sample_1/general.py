import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Grasp Cube A
    if not self.agent.check_grasp(self.cubeA, max_angle=30):
        # Reward for reducing the distance to Cube A
        gripper_pos = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        dist_gripper_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
        reward += -dist_gripper_to_cubeA  # Penalize distance to encourage approach
    else:
        # Milestone 2: Lift Cube A above Cube B
        cubeB_pos = self.cubeB.pose.p
        height_diff = self.cubeA.pose.p[2] - cubeB_pos[2]
        if height_diff < 0.02 * 2:  # If Cube A is not lifted above Cube B
            reward += -0.1  # Penalize for not lifting
        else:
            # Milestone 3: Align Cube A with Cube B
            horizontal_dist = np.linalg.norm(self.cubeA.pose.p[:2] - cubeB_pos[:2])
            if horizontal_dist > 0.02:  # If Cube A is not aligned with Cube B
                reward += -0.1  # Penalize for misalignment
            else:
                # Milestone 4: Place Cube A on Cube B
                placement_height_diff = self.cubeA.pose.p[2] - cubeB_pos[2]
                if np.abs(placement_height_diff - 0.02) > 0.01:  # If Cube A is not placed on Cube B
                    reward += -0.1  # Penalize for incorrect placement
                else:
                    # Milestone 5: Stabilize Cube A
                    if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                        # Task completed: Cube A is stably placed on Cube B
                        reward += 10.0  # Large reward for task completion
                    else:
                        reward += -0.1  # Penalize for instability

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward