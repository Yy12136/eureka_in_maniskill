import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for grasping Cube A
    if not self.robot.check_grasp(self.cubeA, max_angle=30):
        # Penalize distance to Cube A until grasped
        gripper_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
        if gripper_to_cubeA_dist < 0.05:
            reward += 1.0  # Reward for being close to Cube A
    else:
        reward += 2.0  # Reward for successfully grasping Cube A

    # Step 2: Reward for lifting Cube A above Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_height = self.cubeA.pose.p[2]
        cubeB_height = self.cubeB.pose.p[2] + 2 * self.cube_half_size
        if cubeA_height > cubeB_height:
            reward += 1.0  # Reward for lifting Cube A above Cube B

    # Step 3: Reward for placing Cube A on Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        if cubeA_to_cubeB_dist < 0.05:
            reward += 1.0  # Reward for placing Cube A near Cube B

    # Step 4: Reward for stable placement and release
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 3.0  # Reward for stable placement and release

    # Step 5: Penalize excessive action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 6: Penalize excessive joint velocities for smoother movements
    reward += -0.01 * np.linalg.norm(self.robot.qvel)

    return reward