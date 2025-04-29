import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights (total number <= 5)
    weight_grasp = 0.4    # Primary weight for grasping the cube
    weight_lift = 0.5     # Primary weight for lifting the cube
    weight_reach = 0.1    # Secondary weight for reaching the cube
    
    # Initialize reward components (total number <= 5)
    reward_grasp = 0.0    # Reward for successful grasp
    reward_lift = 0.0     # Reward for lifting the cube
    reward_reach = 0.0    # Reward for reaching the cube
    
    # Calculate reward components
    # 1. Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 2. Reward for lifting the cube by 0.2 meters
    cube_height = self.obj.pose.p[2]  # Z-coordinate of the cube
    target_height = 0.2
    if cube_height >= target_height:
        reward_lift = 1.0
    else:
        reward_lift = cube_height / target_height  # Linear scaling
    
    # 3. Reward for reaching the cube (distance between TCP and cube)
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    distance = np.linalg.norm(tcp_pos - cube_pos)
    reward_reach = 1.0 / (1.0 + distance)  # Inverse scaling
    
    # Combine main rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_reach * reward_reach
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= target_height:
        reward += 0.1  # Small bonus for maintaining height
    
    # 2. Penalty for large actions (regularization)
    action_magnitude = sum([a**2 for a in action]) ** 0.5
    reward -= 0.05 * action_magnitude  # Penalize large actions
    
    return reward