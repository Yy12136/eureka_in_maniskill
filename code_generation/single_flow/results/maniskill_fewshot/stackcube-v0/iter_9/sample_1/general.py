import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if not self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is not grasped
        gripper_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
        if gripper_to_cubeA_dist < 0.05:  # If gripper is close to Cube A
            if self.robot.check_grasp(self.cubeA, max_angle=30):  # Successful grasp
                reward += 1.0  # Reward for grasping Cube A
    else:
        # Step 2: Lift Cube A above Cube B
        if self.cubeA.pose.p[2] < self.cubeB.pose.p[2] + 2 * self.cube_half_size:
            reward += 0.5  # Reward for lifting Cube A above Cube B

        # Step 3: Place Cube A on Cube B
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is close to Cube B
            reward += 1.0  # Reward for placing Cube A near Cube B

            # Step 4: Ensure Cube A is stable on Cube B and not grasped
            if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
                reward += 2.0  # Reward for stable placement and release

    # Step 5: Penalize action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    return reward