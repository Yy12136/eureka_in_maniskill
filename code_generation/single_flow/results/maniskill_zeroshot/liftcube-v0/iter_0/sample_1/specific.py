import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant poses and variables
    ee_pose = self.tcp.pose.p  # End-effector position
    cubeA_pose = self.obj.pose.p  # Cube A position
    cubeA_height = cubeA_pose[2]  # Current height of cube A
    initial_height = cubeA_pose[2]  # Initial height of cube A
    target_height = initial_height + self.goal_height  # Target height (0.2 meters above initial height)

    # Stage 1: Approach Cube A
    # Reward for reducing the distance between the gripper and cube A
    distance_to_cubeA = np.linalg.norm(ee_pose - cubeA_pose)
    approach_reward = -distance_to_cubeA  # Negative reward for distance
    reward += approach_reward * 0.5  # Weight for approach stage

    # Stage 2: Grasp Cube A
    # Reward for successfully grasping cube A
    if distance_to_cubeA < 0.05:  # Threshold for being close enough to grasp
        grasp_success = self.agent.check_grasp(self.obj)
        grasp_reward = 1.0 if grasp_success else 0.0
        reward += grasp_reward * 1.0  # Weight for grasp stage

    # Stage 3: Lift Cube A
    # Reward for lifting cube A to the target height
    if self.agent.check_grasp(self.obj):  # Only if cube A is grasped
        height_diff = cubeA_height - initial_height
        lift_reward = -np.abs(height_diff - self.goal_height)  # Negative reward for height error
        reward += lift_reward * 1.0  # Weight for lift stage

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = -np.linalg.norm(action) * 0.01
    reward += action_penalty

    return reward