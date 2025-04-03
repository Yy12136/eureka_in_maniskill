import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    # Reward for being close enough to grasp Cube A
    if gripper_to_cubeA_distance < 0.05:
        reward += 0.2  # Small reward for approaching

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05 and self.robot.gripper_openness < 0.1:
        if self.agent.check_grasp(self.obj, max_angle=30):
            reward += 0.3  # Reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Calculate the height difference between Cube A and the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        # Reward for lifting Cube A close to the goal height
        if height_difference < 0.01:
            reward += 0.5  # Large reward for lifting to the correct height

    # Step 4: Task Completion
    if self.agent.check_grasp(self.obj, max_angle=30) and height_difference < 0.01:
        if check_actor_static(self.obj):
            reward += 1.0  # Final reward for completing the task

    return reward