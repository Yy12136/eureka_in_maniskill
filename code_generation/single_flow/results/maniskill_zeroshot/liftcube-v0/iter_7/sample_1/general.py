import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for being close enough to Cube A
    if gripper_to_cubeA_distance < 0.05:
        reward += 0.2  # Sparse reward for successful approach

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05:
        if self.robot.gripper_openness < 0.1:  # Fully closed gripper
            if self.agent.check_grasp(self.obj, max_angle=30):
                reward += 0.3  # Sparse reward for successful grasp
            else:
                reward -= 0.1  # Small penalty for failed grasp
        else:
            reward -= 0.05  # Small penalty for not closing the gripper

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A to the goal height
        if height_difference < 0.01:
            reward += 0.4  # Sparse reward for successful lift
        else:
            reward -= 0.05  # Small penalty for not reaching the goal height
    else:
        reward -= 0.2  # Penalty for dropping Cube A during lifting

    # Step 4: Stabilize Cube A
    if self.agent.check_grasp(self.obj, max_angle=30) and height_difference < 0.01:
        if check_actor_static(self.obj):
            reward += 0.1  # Sparse reward for stabilization
        else:
            reward -= 0.05  # Small penalty for instability

    return reward