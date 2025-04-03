import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # 1. Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for reaching
        reward += 1.0  # Sparse reward for reaching

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Sparse reward for successful grasp

        # 3. Lifting the cube
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height:  # Goal height is 0.2 meters
            reward += 5.0  # Sparse reward for lifting to the goal height

        # 4. Stabilizing the cube
        is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
        is_cubeA_static = self.cubeA.check_static()
        is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

        if is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
            reward += 10.0  # Sparse reward for successful completion

    # 5. Penalize excessive action magnitude
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward