import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Grasp Cube A
    is_grasped = self.agent.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 1.0

        # Milestone 2: Lift Cube A above Cube B
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        height_diff = cubeA_pos[2] - cubeB_pos[2]
        if height_diff > 0.02 * 2:
            reward += 1.0

            # Milestone 3: Place Cube A on Cube B
            horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
            if horizontal_dist < 0.02:
                reward += 1.0

                # Milestone 4: Stabilize Cube A
                if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                    reward += 1.0

    # Regularization of the robot's action to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward