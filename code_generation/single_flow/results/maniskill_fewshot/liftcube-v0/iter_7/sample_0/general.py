import numpy as np

def compute_sparse_reward(self, action) -> float:
    reward = 0.0

    # 1. Task completion reward
    is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
    is_cubeA_static = self.cubeA.check_static()
    is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

    if is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
        reward += 100.0  # Large reward for task completion
        return reward  # Early termination since the task is complete

    # 2. Intermediate rewards for progress
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        # Reward for lifting the cube towards the goal height
        cubeA_height = self.cubeA.pose.p[2]
        height_diff = self.goal_height - cubeA_height
        lifting_reward = 1 - np.tanh(5.0 * max(height_diff, 0))
        reward += lifting_reward * 10.0  # Encourage lifting progress

    # 3. Penalize excessive joint velocity to encourage smooth movements
    joint_velocity_penalty = -0.02 * np.linalg.norm(self.robot.qvel)
    reward += joint_velocity_penalty

    # 4. Penalize excessive gripper force to encourage gentle grasping
    if is_grasped:
        gripper_force = np.linalg.norm(self.robot.lfinger.pose.p - self.robot.rfinger.pose.p)
        gripper_force_penalty = -0.02 * gripper_force
        reward += gripper_force_penalty

    # 5. Penalize excessive movement of the cube during lifting
    if is_grasped and not is_cubeA_at_goal_height:
        cubeA_velocity = np.linalg.norm(self.cubeA.velocity)
        velocity_penalty = -0.2 * cubeA_velocity
        reward += velocity_penalty

    # 6. Penalize excessive rotation of the cube during lifting
    if is_grasped and not is_cubeA_at_goal_height:
        cubeA_angular_velocity = np.linalg.norm(self.cubeA.angular_velocity)
        angular_velocity_penalty = -0.2 * cubeA_angular_velocity
        reward += angular_velocity_penalty

    return reward