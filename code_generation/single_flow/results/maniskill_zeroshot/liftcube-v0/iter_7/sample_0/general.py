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
        reward += 0.2  # Small reward for approach

    # Step 2: Grasp Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):
        reward += 0.3  # Reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A to the goal height
        if height_difference < 0.01:
            reward += 0.4  # Reward for reaching the goal height

    # Step 4: Stabilize Cube A
    if self.obj.check_static():
        reward += 0.1  # Reward for stabilizing the cube

    # Step 5: Penalize dropping Cube A
    if not self.agent.check_grasp(self.obj, max_angle=30) and self.obj.pose.p[2] > 0.0:
        reward -= 0.5  # Penalty for dropping the cube during lifting

    return reward