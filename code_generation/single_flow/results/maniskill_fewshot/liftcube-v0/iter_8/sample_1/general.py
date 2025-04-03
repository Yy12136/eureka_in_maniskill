import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # 1. Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for reaching the cube
        reward += 1.0  # Reward for reaching the cube

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # 3. Lifting the cube
    if is_grasped:
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height:  # Threshold for lifting the cube
            reward += 3.0  # Reward for lifting the cube to the goal height

    # 4. Stabilizing the cube
    if is_grasped and cubeA_height >= self.goal_height:
        is_cubeA_static = self.cubeA.check_static()
        is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
        if is_cubeA_static and is_robot_static:
            reward += 4.0  # Reward for stabilizing the cube

    # 5. Task completion bonus
    if is_grasped and cubeA_height >= self.goal_height and is_cubeA_static and is_robot_static:
        reward += 10.0  # Large bonus for completing the task

    return reward