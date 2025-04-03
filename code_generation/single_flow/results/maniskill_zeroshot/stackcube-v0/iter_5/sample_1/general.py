import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Grasp Cube A
    if not self.agent.check_grasp(self.cubeA, max_angle=30):
        # Reward for approaching Cube A (only if not yet grasped)
        gripper_pos = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        dist_gripper_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
        reward += 1.0 if dist_gripper_to_cubeA < 0.05 else 0.0
    else:
        # Milestone 2: Lift Cube A above Cube B
        cubeB_pos = self.cubeB.pose.p
        cubeA_pos = self.cubeA.pose.p
        height_diff = cubeA_pos[2] - cubeB_pos[2]
        if height_diff > 0.02 * 2:
            # Milestone 3: Align Cube A with Cube B
            horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
            if horizontal_dist < 0.02:
                # Milestone 4: Place Cube A on Cube B and stabilize
                if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                    # Reward for task completion
                    reward += 10.0

    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward