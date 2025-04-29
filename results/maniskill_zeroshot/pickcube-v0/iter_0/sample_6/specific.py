import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.0500    # Reward for successful grasp
    weight_distance = 0.8000    # Reward for minimizing distance to goal
    weight_action = 0.1500    # Penalty for large actions (regularization)
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0          # Reward for grasping the cube
    reward_distance = 0.0       # Reward for moving the cube closer to the goal
    reward_action = 0.0         # Penalty for large actions
    
    # Calculate reward components
    # 1. Grasp reward (binary reward if cube is grasped)
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Distance reward (negative distance to encourage minimizing distance)
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
    if cube_to_goal_distance < 0.01:  # Threshold for success
        reward += 1.0  # Success bonus
    
    return reward