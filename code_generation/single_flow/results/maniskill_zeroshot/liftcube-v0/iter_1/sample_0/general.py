import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Check if the gripper is close enough to Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    gripper_to_cubeA_distance = np.linalg.norm(ee_pose_wrt_cubeA.p[:2])  # Ignore z-axis for approach

    if gripper_to_cubeA_distance < 0.05:  # Threshold for proximity to grasp
        reward += 0.2  # Reward for approaching Cube A

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_distance < 0.05 and self.robot.gripper_openness < 0.1:  # Fully closed gripper
        if self.agent.check_grasp(self.obj, max_angle=30):
            reward += 0.3  # Reward for successful grasp

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Ensure Cube A is still grasped
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)

        if height_difference < 0.01:  # Threshold for reaching the goal height
            reward += 0.5  # Reward for lifting Cube A to the goal height

    # Step 4: Stabilize Cube A
    if height_difference < 0.01 and check_actor_static(self.obj):  # Check if Cube A is stable
        reward += 0.2  # Reward for stabilizing Cube A

    # Step 5: Task Completion Bonus
    if height_difference < 0.01 and check_actor_static(self.obj):
        reward += 1.0  # Large bonus for completing the task

    # Step 6: Penalize Large Actions (Optional)
    # Encourage smooth movements by penalizing large actions
    action_penalty = -np.linalg.norm(action) * 0.1
    reward += action_penalty

    return reward