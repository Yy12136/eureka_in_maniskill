import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.3       # Reward for successful grasping
    weight_distance = 0.5    # Reward for minimizing distance to goal
    weight_action = 0.2      # Regularization for action smoothness
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0       # Reward for successful grasp
    reward_distance = 0.0    # Reward for minimizing distance to goal
    reward_action = 0.0      # Regularization for action smoothness
    
    # Calculate reward components
    # 1. Grasp reward (binary reward for successful grasp)
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Distance reward (negative distance to encourage minimizing distance)
    if self.agent.check_grasp(self.obj):
        # Distance between cube and goal position
        distance = np.linalg.norm(self.obj.pose.p - self.goal_pos)
        reward_distance = -distance  # Negative distance to encourage minimizing it
    else:
        # Distance between gripper and cube (before grasping)
        distance = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
        reward_distance = -distance  # Negative distance to encourage minimizing it
    
    # 3. Action regularization (penalize large actions for smooth control)
    reward_action = -np.square(action).sum()
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_distance * reward_distance +
        weight_action * reward_action
    )
    
    # Optional: success bonus (if cube is at the goal position)
    if np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:  # Threshold for success
        reward += 1.0  # Success bonus
    
    return reward