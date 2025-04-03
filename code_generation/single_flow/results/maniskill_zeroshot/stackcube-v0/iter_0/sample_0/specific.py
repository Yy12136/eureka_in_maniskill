import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and Cube A
    gripper_pose = self.tcp.pose.p
    cubeA_pos = self.cubeA.pose.p
    dist_to_cubeA = np.linalg.norm(gripper_pose - cubeA_pos)
    reward += -dist_to_cubeA * 0.5  # Encourage the gripper to move closer to Cube A

    # Step 2: Grasp Cube A
    if dist_to_cubeA < 0.05:  # If the gripper is close enough to Cube A
        is_grasped = self.agent.check_grasp(self.cubeA, max_angle=30)
        if is_grasped:
            reward += 1.0  # Reward for successful grasp
        else:
            reward += -0.5  # Penalize for not grasping

    # Step 3: Lift Cube A
    if self.agent.check_grasp(self.cubeA, max_angle=30):
        # Calculate the height difference between Cube A and Cube B
        cubeB_pos = self.cubeB.pose.p
        height_diff = cubeA_pos[2] - cubeB_pos[2]
        if height_diff < 0.1:  # Cube A should be lifted above Cube B
            reward += -0.5  # Penalize if Cube A is not lifted high enough
        else:
            reward += 0.5  # Reward for lifting Cube A

    # Step 4: Move Cube A Over Cube B
    if self.agent.check_grasp(self.cubeA, max_angle=30) and height_diff >= 0.1:
        # Calculate the horizontal distance between Cube A and Cube B
        horizontal_dist = np.linalg.norm(cubeA_pos[:2] - cubeB_pos[:2])
        reward += -horizontal_dist * 0.5  # Encourage Cube A to be directly above Cube B

    # Step 5: Place Cube A on Cube B
    if self.agent.check_grasp(self.cubeA, max_angle=30) and horizontal_dist < 0.05:
        # Lower Cube A onto Cube B
        if cubeA_pos[2] - cubeB_pos[2] < 0.02:  # Cube A is placed on Cube B
            reward += 1.0  # Reward for placing Cube A on Cube B
        else:
            reward += -0.5  # Penalize for not lowering Cube A

    # Step 6: Verify Stability
    if not self.agent.check_grasp(self.cubeA, max_angle=30) and cubeA_pos[2] - cubeB_pos[2] < 0.02:
        if check_actor_static(self.cubeA):  # Cube A is static on Cube B
            reward += 2.0  # Reward for stable placement
        else:
            reward += -1.0  # Penalize if Cube A is not static

    # Regularization of the robot's action
    action_penalty = np.linalg.norm(action) * 0.01
    reward -= action_penalty  # Penalize large actions to encourage smooth movements

    return reward