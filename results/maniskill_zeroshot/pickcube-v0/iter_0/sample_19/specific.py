import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.3         # Weight for successful grasp
    weight_distance = 0.5      # Weight for minimizing distance to goal
    weight_action = 0.2        # Weight for action regularization
    # Note: weight_grasp + weight_distance + weight_action = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0         # Reward for successful grasp
    reward_distance = 0.0      # Reward for minimizing distance to goal
    reward_action = 0.0        # Reward for action regularization
    
    # Calculate reward components
    # 1. Grasp reward (binary reward for successful grasp)
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Distance reward (negative distance to encourage minimizing distance)
    # Distance between cube and goal position
    cube_to_goal_distance = np.linalg.norm(self.obj.pose.p - self.goal_pos)
    reward_distance = -cube_to_goal_distance
    
    # 3. Action regularization (penalize large actions)
    reward_action = -np.square(action).sum()
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_distance * reward_distance +
        weight_action * reward_action
    )
    
    # Optional: success bonus (if cube is at the goal position)
    if cube_to_goal_distance < 0.01:  # Threshold for task completion
        reward += 1.0  # Success bonus
    
    return reward