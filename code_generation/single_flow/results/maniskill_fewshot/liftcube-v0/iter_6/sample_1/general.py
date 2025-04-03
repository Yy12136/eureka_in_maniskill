import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for reaching the cube
        reward += 1.0  # Sparse reward for reaching the cube

    # 2. Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Sparse reward for successful grasp

        # 3. Lifting the cube
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height - 0.01:  # Threshold for lifting the cube
            reward += 3.0  # Sparse reward for lifting the cube to the goal height

        # 4. Stabilizing the cube
        is_cubeA_static = self.cubeA.check_static()
        is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
        if is_cubeA_static and is_robot_static:
            reward += 4.0  # Sparse reward for stabilizing the cube and robot

    # 5. Penalize excessive joint velocity of the robot
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.robot.qvel)
    reward += joint_velocity_penalty

    # 6. Penalize excessive gripper force when grasping
    if is_grasped:
        gripper_force = np.linalg.norm(self.robot.lfinger.pose.p - self.robot.rfinger.pose.p)
        gripper_force_penalty = -0.01 * gripper_force
        reward += gripper_force_penalty

    return reward