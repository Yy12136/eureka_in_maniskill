import numpy as np

def compute_sparse_reward(self) -> float:
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
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height:  # Threshold for lifting the cube
            reward += 5.0  # Reward for lifting the cube to the goal height

    # 4. Task completion (cube is lifted and stable)
    is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
    is_cubeA_static = self.cubeA.check_static()
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    if is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
        reward += 10.0  # Reward for successful task completion

    return reward