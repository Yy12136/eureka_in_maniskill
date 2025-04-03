import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    """
    Compute a dense reward for the task: Pick up cube A and lift it up by 0.2 meters.
    The reward is staged and consists of:
    1. Distance between the robot's end-effector and cube A.
    2. Successful grasp of cube A.
    3. Lifting cube A to the target height.
    """
    reward = 0.0  # Initialize the reward

    # Step 1: Approach Cube A
    # Calculate the distance between the end-effector and cube A
    ee_pose = self.tcp.pose.p
    cubeA_pose = self.obj.pose.p
    distance_to_cubeA = np.linalg.norm(ee_pose - cubeA_pose)
    # Reward for reducing the distance to cube A
    reward += max(0, 1.0 - distance_to_cubeA) * 0.3

    # Step 2: Grasp Cube A
    # Check if the robot is grasping cube A
    is_grasping = self.agent.check_grasp(self.obj, max_angle=30)
    if is_grasping:
        # Reward for successfully grasping cube A
        reward += 0.4

    # Step 3: Lift Cube A
    if is_grasping:
        # Calculate the height difference between cube A and the target height
        cubeA_height = self.obj.pose.p[2]  # Z-axis position
        height_difference = abs(cubeA_height - self.goal_height)
        # Reward for lifting cube A closer to the target height
        reward += max(0, 1.0 - height_difference) * 0.3

    # Step 4: Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.1 * np.linalg.norm(action)
    reward += action_penalty

    return reward