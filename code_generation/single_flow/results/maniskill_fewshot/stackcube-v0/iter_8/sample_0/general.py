import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Reward for successful grasp of Cube A
    if not self.robot.check_grasp(self.cubeA, max_angle=30):
        # Penalize if Cube A is not grasped
        reward += -0.1
    else:
        # Reward for successful grasp
        reward += 1.0

    # Step 2: Reward for lifting Cube A above Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        if self.cubeA.pose.p[2] >= self.cubeB.pose.p[2] + 2 * self.cube_half_size:
            reward += 1.0  # Reward for lifting Cube A

    # Step 3: Reward for placing Cube A on Cube B stably
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        # Check if Cube A is on top of Cube B
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
        if horizontal_dist < 0.01:  # Cube A is centered on Cube B
            reward += 2.0  # Reward for stable placement

    # Step 4: Penalize for unnecessary actions
    reward += -0.01 * np.linalg.norm(action)  # Penalize large actions for smoother movements

    # Step 5: Penalize for excessive movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    return reward