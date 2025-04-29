import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_distance_to_cube = 0.4  # Encourage moving gripper closer to cube
    weight_distance_to_goal = 0.4  # Encourage moving cube closer to goal
    weight_grasp = 0.1            # Reward successful grasp
    weight_action_regularization = 0.1  # Penalize large actions for stability

    # Initialize reward components (total number <= 5)
    reward_distance_to_cube = 0.0  # Distance between gripper and cube
    reward_distance_to_goal = 0.0  # Distance between cube and goal
    reward_grasp = 0.0             # Binary reward for successful grasp
    reward_action_regularization = 0.0  # Penalty for large actions

    # Calculate reward components
    # 1. Distance between gripper and cube (negative to minimize distance)
    gripper_to_cube_dist = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    reward_distance_to_cube = -gripper_to_cube_dist

    # 2. Distance between cube and goal (negative to minimize distance)
    cube_to_goal_dist = np.linalg.norm(self.obj.pose.p - self.goal_pos)
    reward_distance_to_goal = -cube_to_goal_dist

    # 3. Binary reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0

    # 4. Penalize large actions for stability
    reward_action_regularization = -np.square(action).sum()

    # Combine main rewards
    reward = (
        weight_distance_to_cube * reward_distance_to_cube +
        weight_distance_to_goal * reward_distance_to_goal +
        weight_grasp * reward_grasp +
        weight_action_regularization * reward_action_regularization
    )

    # Optional: success bonus if cube is at goal
    if cube_to_goal_dist < 0.01:  # Threshold for success
        reward += 10.0  # Large bonus for task completion

    return reward