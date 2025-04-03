import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Reaching the cube (optional, can be removed for true sparsity)
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for proximity to the cube
        reward += 1.0  # Small reward for reaching the cube

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 5.0  # Significant reward for successful grasp

        # 3. Lifting the cube to the goal height
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height - 0.01:  # Threshold for reaching the goal height
            reward += 10.0  # Large reward for task completion

    # 4. Penalize dropping the cube (optional)
    if not is_grasped and self.cubeA.pose.p[2] > 0.0:  # Cube is in the air but not grasped
        reward -= 5.0  # Penalty for dropping the cube

    return reward