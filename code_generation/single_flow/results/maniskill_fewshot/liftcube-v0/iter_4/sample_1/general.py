import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Reaching the cube (optional, can be removed for fully sparse reward)
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for reaching
        reward += 1.0  # Small reward for reaching the cube

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

    # 3. Lifting the cube to the goal height
    cubeA_height = self.cubeA.pose.p[2]
    if is_grasped and cubeA_height >= self.goal_height:
        reward += 10.0  # Large reward for lifting the cube to the goal height

    # 4. Stabilizing the cube at the goal height
    if is_grasped and cubeA_height >= self.goal_height:
        is_cubeA_static = self.cubeA.check_static()
        is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
        if is_cubeA_static and is_robot_static:
            reward += 15.0  # Final reward for successful completion

    return reward