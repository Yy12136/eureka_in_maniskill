import numpy as np

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # 1. Task completion reward (sparse)
    is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
    is_cubeA_static = self.cubeA.check_static()
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    if is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
        reward += 100.0  # Large reward for successful task completion
        return reward  # Terminate early since the task is complete

    # 2. Grasping reward (sparse)
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 10.0  # Reward for successful grasp

        # 3. Lifting progress reward (dense but sparse-like)
        cubeA_height = self.cubeA.pose.p[2]
        height_diff = self.goal_height - cubeA_height
        if height_diff > 0:
            reward += 5.0 * (1 - np.tanh(5.0 * height_diff))  # Reward proportional to progress

    # 4. Penalize excessive movement or instability
    if is_grasped:
        cubeA_velocity = np.linalg.norm(self.cubeA.velocity)
        cubeA_angular_velocity = np.linalg.norm(self.cubeA.angular_velocity)
        reward -= 0.5 * (cubeA_velocity + cubeA_angular_velocity)  # Penalize instability

    # 5. Regularization of the robot's action
    reward -= 0.01 * np.linalg.norm(action)  # Penalize large actions

    return reward