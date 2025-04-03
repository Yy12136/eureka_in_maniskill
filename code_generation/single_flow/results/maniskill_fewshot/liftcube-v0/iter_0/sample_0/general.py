import numpy as np

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Goal height for cube A
    goal_height = self.cubeA.pose.p[2] + 0.2  # Lift cube A by 0.2 meters

    # Stage 1: Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Encourage reaching the cube
        reward += 1.0

    # Stage 2: Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Stage 3: Lifting the cube
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= goal_height:  # Reward for lifting to the goal height
            reward += 5.0

            # Stage 4: Stabilizing the cube
            is_cubeA_static = self.cubeA.check_static()
            is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
            if is_cubeA_static and is_robot_static:  # Reward for stable completion
                reward += 10.0

    # Penalize excessive gripper openness when not grasping
    if not is_grasped and self.robot.gripper_openness < 0.2:
        reward -= 0.5

    # Penalize excessive movement of the cube when lifting
    if is_grasped and cubeA_height < goal_height:
        cubeA_velocity = np.linalg.norm(self.cubeA.velocity)
        reward -= 0.1 * cubeA_velocity

    # Penalize excessive rotation of the cube when lifting
    if is_grasped and cubeA_height < goal_height:
        cubeA_angular_velocity = np.linalg.norm(self.cubeA.angular_velocity)
        reward -= 0.1 * cubeA_angular_velocity

    # Penalize excessive joint velocity of the robot
    joint_velocity_penalty = np.linalg.norm(self.robot.qvel)
    reward -= 0.01 * joint_velocity_penalty

    return reward