import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.36130659286442934
    weight_lift = 0.27738681427116363
    weight_static = 0.3613065928644071
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static during lift
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Higher reward if closer to the cube
    
    # Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        current_height = cube_pos[2] - 0.02
        height_diff = target_height - current_height
        reward_lift = 1.0 - np.tanh(np.abs(height_diff))  # Higher reward if closer to target height
    
    # Reward for keeping the cube static during lift
    if self.agent.check_grasp(self.obj) and check_actor_static(self.obj):
        reward_static = 1.0  # Full reward if the cube is static
    
    # Combine all rewards
    reward = weight_grasp * reward_grasp + weight_lift * reward_lift + weight_static * reward_static
    
    return reward