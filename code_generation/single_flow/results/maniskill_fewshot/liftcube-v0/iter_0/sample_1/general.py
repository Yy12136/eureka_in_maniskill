import numpy as np

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Goal height for lifting the cube
    goal_height = self.cubeA.pose.p[2] + 0.2

    # Stage 1: Reaching the cube
    tcp_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    if tcp_to_cubeA_dist < 0.01:  # Sparse reward for reaching the cube
        reward += 1.0

    # Stage 2: Grasping the cube
    is_grasped = self.robot.check_grasp(self.cubeA, max_angle=30)
    if is_grasped:  # Sparse reward for successful grasp
        reward += 2.0

        # Stage 3: Lifting the cube
        cubeA_height = self.cubeA.pose.p[2]
        if cubeA_height >= goal_height:  # Sparse reward for reaching the goal height
            reward += 5.0

            # Stage 4: Stabilizing the cube and robot
            is_cubeA_static = self.cubeA.check_static()
            is_robot_static = np.max(np.abs(self.robot.qvel)) <= 0.2
            if is_cubeA_static and is_robot_static:  # Sparse reward for successful completion
                reward += 10.0

    return reward