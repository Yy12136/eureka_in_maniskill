import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Reaching the cube (only reward when close enough)
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for being close to the cube
        reward += 1.0

    # 2. Grasping the cube (binary reward for successful grasp)
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0

        # 3. Lifting the cube (binary reward for reaching the goal height)
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height - 0.01:  # Threshold for reaching the goal height
            reward += 5.0

            # 4. Stabilizing the cube (binary reward for completing the task)
            is_cubeA_static = self.cubeA.check_static()
            is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
            if is_cubeA_static and is_robot_static:
                reward += 10.0  # Large reward for successful completion

    return reward