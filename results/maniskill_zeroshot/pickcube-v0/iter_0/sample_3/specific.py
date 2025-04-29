import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_distance_to_cube = 0.0750    # Encourage moving gripper closer to cube
    weight_distance_to_goal = 0.0750    # Encourage moving cube closer to goal
    weight_grasp = 0.8000    # Reward successful grasping
    weight_action_regularization = 0.0500    # Penalize large actions for stability
    
    # Initialize reward components (total number <= 5)
    reward_distance_to_cube = 0.0    # Distance between gripper and cube
    reward_distance_to_goal = 0.0    # Distance between cube and goal
    reward_grasp = 0.0               # Binary reward for grasping
    reward_action_regularization = 0.0  # Penalty for large actions
    
    # Calculate reward components
    # 1. Distance between gripper and cube (negative to minimize distance)
    reward_distance_to_cube = -np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    
    # 2. Distance between cube and goal (negative to minimize distance)
    reward_distance_to_goal = -np.linalg.norm(self.obj.pose.p - self.goal_pos)
    
    # 3. Binary reward for successful grasp
    reward_grasp = 1.0 if self.agent.check_grasp(self.obj) else 0.0
    
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
    if np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:  # Threshold for success
        reward += 1.0  # Success bonus
    
    return reward