import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_distance_to_cube = 0.3  # Encourage moving gripper closer to the cube
    weight_distance_to_goal = 0.5  # Encourage moving the cube closer to the goal
    weight_grasp = 0.1             # Reward successful grasping
    weight_action_regularization = 0.1  # Penalize large actions for stability

    # Initialize reward components (total number <= 5)
    reward_distance_to_cube = 0.0  # Distance between gripper and cube
    reward_distance_to_goal = 0.0  # Distance between cube and goal
    reward_grasp = 0.0             # Grasp state
    reward_action_regularization = 0.0  # Action regularization

    # Calculate reward components
    # 1. Distance between gripper and cube (negative to minimize distance)
    gripper_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    reward_distance_to_cube = -np.linalg.norm(gripper_pos - cube_pos)

    # 2. Distance between cube and goal (negative to minimize distance)
    goal_pos = self.goal_pos
    reward_distance_to_goal = -np.linalg.norm(cube_pos - goal_pos)

    # 3. Grasp state (binary reward for successful grasp)
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0

    # 4. Action regularization (penalize large actions)
    reward_action_regularization = -np.square(action).sum()

    # Combine main rewards
    reward = (
        weight_distance_to_cube * reward_distance_to_cube +
        weight_distance_to_goal * reward_distance_to_goal +
        weight_grasp * reward_grasp +
        weight_action_regularization * reward_action_regularization
    )

    # Optional: success bonus (if cube is at the goal position)
    if np.linalg.norm(cube_pos - goal_pos) < 0.01:  # Threshold for success
        reward += 10.0  # Large bonus for task completion

    return reward