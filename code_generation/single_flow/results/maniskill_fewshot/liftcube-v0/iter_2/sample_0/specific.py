import numpy as np

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Stage 1: Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    reaching_reward = 1 - np.tanh(5.0 * tcp_to_cubeA_dist)
    reward += reaching_reward * 0.4  # Encourage reaching the cube

    # Stage 2: Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:
        reward += 2.0  # Reward for successful grasp

        # Stage 3: Lifting the cube
        cubeA_height = self.cubeA.pose.p[2]
        height_diff = self.goal_height - cubeA_height
        lifting_reward = 1 - np.tanh(5.0 * max(height_diff, 0))  # Reward for lifting towards the goal height
        reward += lifting_reward * 0.6  # Encourage lifting

        # Stage 4: Stabilizing the cube
        is_cubeA_at_goal_height = np.abs(self.cubeA.pose.p[2] - self.goal_height) <= 0.01
        is_cubeA_static = self.cubeA.check_static()
        is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2

        if is_cubeA_at_goal_height and is_cubeA_static and is_robot_static:
            reward += 15.0  # Reward for successful completion

    # Regularization of the robot's action
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    # Penalize excessive gripper openness when not grasping
    if not is_grasped and self.robot.gripper_openness < 0.2:
        gripper_penalty = -0.2 * (0.2 - self.robot.gripper_openness)
        reward += gripper_penalty

    # Penalize excessive movement of the cube when lifting
    if is_grasped and not is_cubeA_at_goal_height:
        cubeA_velocity = np.linalg.norm(self.cubeA.velocity)
        velocity_penalty = -0.2 * cubeA_velocity
        reward += velocity_penalty

    # Penalize excessive rotation of the cube when lifting
    if is_grasped and not is_cubeA_at_goal_height:
        cubeA_angular_velocity = np.linalg.norm(self.cubeA.angular_velocity)
        angular_velocity_penalty = -0.2 * cubeA_angular_velocity
        reward += angular_velocity_penalty

    # Encourage smooth lifting by penalizing sudden changes in height
    if is_grasped and not is_cubeA_at_goal_height:
        height_change = np.abs(self.cubeA.velocity[2])
        height_change_penalty = -0.2 * height_change
        reward += height_change_penalty

    # Penalize excessive joint velocity of the robot
    joint_velocity_penalty = -0.02 * np.linalg.norm(self.robot.qvel)
    reward += joint_velocity_penalty

    # Penalize excessive gripper velocity when not grasping
    if not is_grasped:
        gripper_velocity_penalty = -0.02 * np.linalg.norm(self.robot.gripper_openness)
        reward += gripper_velocity_penalty

    # Encourage proper alignment of the gripper with the cube
    if not is_grasped:
        gripper_to_cubeA_angle = self.robot.ee_pose.q.dot(self.cubeA.pose.q)
        alignment_reward = 1 - np.tanh(5.0 * (1 - gripper_to_cubeA_angle))
        reward += alignment_reward * 0.3  # Encourage proper alignment

    # Encourage proper orientation of the cube during lifting
    if is_grasped and not is_cubeA_at_goal_height:
        cubeA_orientation_diff = np.linalg.norm(self.cubeA.pose.q - np.array([1, 0, 0, 0]))  # Desired orientation is [1, 0, 0, 0]
        orientation_penalty = -0.2 * cubeA_orientation_diff
        reward += orientation_penalty

    # Penalize excessive gripper force when grasping
    if is_grasped:
        gripper_force = np.linalg.norm(self.robot.lfinger.pose.p - self.robot.rfinger.pose.p)
        gripper_force_penalty = -0.02 * gripper_force
        reward += gripper_force_penalty

    # Encourage smooth joint movements
    joint_acceleration_penalty = -0.01 * np.linalg.norm(np.diff(self.robot.qvel))
    reward += joint_acceleration_penalty

    # Encourage smooth gripper movements
    if not is_grasped:
        gripper_acceleration_penalty = -0.01 * np.linalg.norm(np.diff(self.robot.gripper_openness))
        reward += gripper_acceleration_penalty

    # Encourage minimal deviation from the vertical path during lifting
    if is_grasped and not is_cubeA_at_goal_height:
        horizontal_deviation = np.linalg.norm(self.cubeA.pose.p[:2] - self.robot.ee_pose.p[:2])
        deviation_penalty = -0.2 * horizontal_deviation
        reward += deviation_penalty

    # Encourage minimal rotation of the gripper during lifting
    if is_grasped and not is_cubeA_at_goal_height:
        gripper_rotation = np.linalg.norm(self.robot.ee_pose.q - np.array([1, 0, 0, 0]))  # Desired orientation is [1, 0, 0, 0]
        rotation_penalty = -0.2 * gripper_rotation
        reward += rotation_penalty

    return reward