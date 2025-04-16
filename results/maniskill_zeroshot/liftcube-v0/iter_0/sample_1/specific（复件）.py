import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_dist = 0.3    # Weight for distance to cube
    weight_grasp = 0.3   # Weight for successful grasp
    weight_lift = 0.3    # Weight for lifting progress
    weight_smooth = 0.1  # Weight for motion smoothness
    # Note: weight_dist + weight_grasp + weight_lift + weight_smooth = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_dist = 0.0    # Distance to cube reward
    reward_grasp = 0.0   # Grasp reward
    reward_lift = 0.0    # Lift progress reward
    reward_smooth = 0.0  # Motion smoothness reward
    
    # Get cube and end-effector positions
    cube_pos = self.obj.pose.p
    ee_pos = self.tcp.pose.p
    
    # Calculate distance to cube
    dist_to_cube = np.linalg.norm(cube_pos - ee_pos)
    reward_dist = max(0, 1 - dist_to_cube / 0.1)  # Normalized to 0-1 range
    
    # Check if cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate lift progress
    target_height = 0.2
    current_height = cube_pos[2]  # Z-coordinate of cube
    reward_lift = max(0, min(current_height / target_height, 1.0))
    
    # Calculate motion smoothness (penalize large actions)
    action_magnitude = np.linalg.norm(action)
    reward_smooth = max(0, 1 - action_magnitude / 2.0)  # Normalized to 0-1 range
    
    # Combine main rewards
    reward = (
        weight_dist * reward_dist +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_smooth * reward_smooth
    )
    
    # Optional: Bonus for maintaining cube above goal height
    if current_height >= target_height:
        reward += 0.1  # Small bonus for achieving the goal
    
    # Optional: Penalty for large actions (regularization)
    if action_magnitude > 2.0:
        reward -= 0.05  # Small penalty for excessive actions
    
    return reward