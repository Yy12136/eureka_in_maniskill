import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Milestone 1: Grasp Cube A
    if not self.agent.check_grasp(self.cubeA, max_angle=30):  # If Cube A is not grasped
        # Reward for approaching Cube A
        gripper_pos = self.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        dist_gripper_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
        if dist_gripper_to_cubeA < 0.05:  # If the gripper is close enough to Cube A
            # Reward for successfully grasping Cube A
            if self.agent.check_grasp(self.cubeA, max_angle=30):
                reward += 1.0
    else:
        # Milestone 2: Lift Cube A above Cube B
        cubeB_pos = self.cubeB.pose.p
        height_diff = self.cubeA.pose.p[2] - cubeB_pos[2]
        if height_diff > 0.02 * 2:  # If Cube A is lifted above Cube B
            reward += 1.0

            # Milestone 3: Place Cube A on Cube B
            horizontal_dist = np.linalg.norm(self.cubeA.pose.p[:2] - cubeB_pos[:2])
            if horizontal_dist < 0.02:  # If Cube A is aligned with Cube B
                reward += 1.0

                # Milestone 4: Stabilize Cube A
                if check_actor_static(self.cubeA) and not self.agent.check_grasp(self.cubeA, max_angle=30):
                    reward += 1.0  # Large reward for task completion

    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward