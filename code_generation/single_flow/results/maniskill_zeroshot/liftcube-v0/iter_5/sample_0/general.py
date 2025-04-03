import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for being close enough to attempt a grasp
    if gripper_to_cubeA_distance < 0.05:
        reward += 0.2  # Sparse reward for approaching

    # Step 2: Grasp Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Check if Cube A is grasped
        reward += 0.3  # Sparse reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A to the goal height
        if height_difference < 0.01:  # Threshold for reaching the goal height
            reward += 0.5  # Sparse reward for successful lift

    # Step 4: Stabilize Cube A
    if self.agent.check_grasp(self.obj, max_angle=30) and height_difference < 0.01:
        if check_actor_static(self.obj):  # Check if Cube A is stable
            reward += 0.2  # Sparse reward for stabilization

    # Step 5: Penalize dropping Cube A
    if not self.agent.check_grasp(self.obj, max_angle=30):
        reward -= 0.5  # Sparse penalty for dropping Cube A

    return reward