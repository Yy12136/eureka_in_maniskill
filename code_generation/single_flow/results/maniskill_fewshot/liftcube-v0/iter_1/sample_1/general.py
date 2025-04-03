import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Task completion reward (sparse)
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
    is_cubeA_static = self.cubeA.check_static()
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    if is_grasped and is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
        reward += 100.0  # Large reward for successful task completion
        return reward  # Early termination since the task is complete

    # 2. Intermediate rewards (sparse but meaningful)
    if is_grasped:
        # Reward for lifting the cube to the goal height
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= self.goal_height:
            reward += 20.0  # Reward for reaching the goal height
        else:
            reward += 5.0  # Small reward for lifting the cube (intermediate progress)
    else:
        # Reward for reaching the cube (pre-grasp)
        tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
        if tcp_to_cubeA_dist <= 0.05:  # Threshold for being close to the cube
            reward += 10.0  # Reward for reaching the cube

    # 3. Penalties for undesirable behaviors (sparse)
    if is_grasped and not is_cubeA_at_goal_height:
        # Penalize excessive movement or rotation of the cube
        cubeA_velocity = np.linalg.norm(self.cubeA.velocity)
        cubeA_angular_velocity = np.linalg.norm(self.cubeA.angular_velocity)
        if cubeA_velocity > 0.1 or cubeA_angular_velocity > 0.1:
            reward -= 5.0  # Penalty for unstable lifting

    # 4. Regularization of the robot's action (sparse)
    if np.linalg.norm(action) > 1.0:  # Threshold for excessive action
        reward -= 2.0  # Penalty for large actions

    return reward