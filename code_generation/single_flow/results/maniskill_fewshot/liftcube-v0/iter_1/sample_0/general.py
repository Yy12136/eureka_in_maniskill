import numpy as np

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Stage 1: Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    if tcp_to_cubeA_dist < 0.05:  # Threshold for reaching the cube
        reward += 1.0  # Sparse reward for reaching the cube

    # Stage 2: Grasping the cube
    is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
    if is_grasped:
        reward += 2.0  # Sparse reward for successful grasp

        # Stage 3: Lifting the cube
        cubeA_height = self.obj.pose.p[2]
        height_diff = self.goal_height - cubeA_height
        if height_diff <= 0.01:  # Threshold for reaching the goal height
            reward += 5.0  # Sparse reward for lifting the cube to the goal height

            # Stage 4: Stabilizing the cube
            is_cubeA_static = check_actor_static(self.obj)
            is_robot_static = np.max(np.abs(self.agent.robot.get_qvel()[:-2])) <= 0.2
            if is_cubeA_static and is_robot_static:
                reward += 10.0  # Sparse reward for successful completion

    # Penalize excessive joint velocity of the robot
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward += joint_velocity_penalty

    # Penalize excessive gripper velocity when not grasping
    if not is_grasped:
        gripper_velocity_penalty = -0.01 * np.linalg.norm(self.robot.gripper_openness)
        reward += gripper_velocity_penalty

    return reward