import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for being close enough to Cube A to attempt a grasp
    if gripper_to_cubeA_distance < 0.05:
        reward += 0.2  # Small reward for successful approach

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05 and self.robot.gripper_openness < 0.1:  # Gripper is closed
        if self.agent.check_grasp(self.obj, max_angle=30):  # Successful grasp
            reward += 0.3  # Moderate reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A to the desired height
        if height_difference < 0.01:  # Threshold for reaching the goal height
            reward += 0.3  # Moderate reward for successful lift

    # Step 4: Stabilize Cube A
    if self.agent.check_grasp(self.obj, max_angle=30) and height_difference < 0.01:
        if check_actor_static(self.obj):  # Cube A is stable
            reward += 0.2  # Small reward for stabilization

    # Step 5: Task Completion Bonus
    # If all steps are completed successfully, provide a large final reward
    if (gripper_to_cubeA_distance < 0.05 and self.robot.gripper_openness < 0.1 and
        self.agent.check_grasp(self.obj, max_angle=30) and height_difference < 0.01 and
        check_actor_static(self.obj)):
        reward += 1.0  # Large reward for task completion

    # Step 6: Penalize Dropping Cube A
    if not self.agent.check_grasp(self.obj, max_angle=30) and height_difference > 0.01:
        reward -= 0.5  # Penalty for dropping Cube A during lifting

    return reward