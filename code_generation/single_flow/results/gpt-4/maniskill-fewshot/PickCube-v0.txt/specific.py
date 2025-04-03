import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get the current pose of cube A and the goal position
    cubeA_pos = self.obj.pose.p
    goal_pos = self.goal_pos

    # Get the end-effector pose
    ee_pos = self.tcp.pose.p

    # Stage 1: Approach cube A
    # Calculate the distance between the end-effector and cube A
    dist_to_cubeA = np.linalg.norm(ee_pos - cubeA_pos)
    # Reward for reducing the distance to cube A
    approach_reward = -dist_to_cubeA
    reward += approach_reward * 0.5  # Weight for approach reward

    # Stage 2: Grasp cube A
    if dist_to_cubeA < 0.05:  # If the gripper is close enough to cube A
        # Check if the robot is grasping cube A
        if self.agent.check_grasp(self.obj, max_angle=30):
            grasp_reward = 1.0
            reward += grasp_reward * 0.3  # Weight for grasp reward

    # Stage 3: Lift cube A
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Calculate the height of cube A
        cubeA_height = cubeA_pos[2]
        # Reward for lifting cube A off the ground
        lift_reward = cubeA_height
        reward += lift_reward * 0.2  # Weight for lift reward

    # Stage 4: Move cube A to the goal position
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Calculate the distance between cube A and the goal position
        dist_to_goal = np.linalg.norm(cubeA_pos - goal_pos)
        # Reward for reducing the distance to the goal position
        move_reward = -dist_to_goal
        reward += move_reward * 0.4  # Weight for move reward

    # Stage 5: Final placement
    if not self.agent.check_grasp(self.obj, max_angle=30):
        # Check if cube A is at the goal position
        if np.linalg.norm(cubeA_pos - goal_pos) < 0.02:  # Tolerance for final placement
            final_reward = 1.0
            reward += final_reward * 0.5  # Weight for final reward

    # Regularization: Penalize large actions
    action_penalty = -np.linalg.norm(action) * 0.01
    reward += action_penalty

    return reward