import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Reaching the cube (optional, can be removed for sparsity)
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for being close to the cube
        reward += 1.0  # Small reward for reaching the cube

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 5.0  # Significant reward for successful grasp

        # 3. Lifting the cube to the goal height
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height - 0.01:  # Tolerance for goal height
            reward += 10.0  # Large reward for task completion

    # 4. Penalize excessive action magnitude (optional, for smoothness)
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward