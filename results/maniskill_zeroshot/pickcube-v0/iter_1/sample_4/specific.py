import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4          # Weight for successful grasp
    weight_distance_to_goal = 0.4  # Weight for minimizing distance to goal
    weight_distance_to_cube = 0.2  # Weight for minimizing distance to cube
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0          # Reward for successful grasp
    reward_distance_to_goal = 0.0  # Reward for minimizing distance to goal
    reward_distance_to_cube = 0.0  # Reward for minimizing distance to cube
    
    # Calculate reward components
    # 1. Grasp reward (binary reward for successful grasp)
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Distance to goal reward (negative distance to encourage minimizing it)
    distance_to_goal = np.linalg.norm(self.obj.pose.p - self.goal_pos)
    reward_distance_to_goal = -distance_to_goal
    
    # 3. Distance to cube reward (negative distance to encourage approaching the cube)
    distance_to_cube = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
    reward_distance_to_cube = -distance_to_cube
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_distance_to_goal * reward_distance_to_goal +
        weight_distance_to_cube * reward_distance_to_cube
    )
    
    # Optional: success bonus (if cube is at the goal position)
    if distance_to_goal < 0.01:  # Threshold for success
        reward += 1.0  # Success bonus
    
    return reward