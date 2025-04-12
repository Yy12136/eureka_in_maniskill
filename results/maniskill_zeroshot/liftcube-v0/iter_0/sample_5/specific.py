import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.14999999999968253
    weight_lift = 0.7999999999956202
    weight_static = 0.05000000000469708
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static
    
    # Get positions
    tcp_pos = self.tcp.pose.p  # TCP position
    cube_pos = self.obj.pose.p  # Cube position
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Higher reward for closer grasp
    
    # Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = 0.2  # Target lift height
        current_height = cube_pos[2]  # Current height of the cube
        height_diff = abs(target_height - current_height)
        reward_lift = 1.0 - np.tanh(height_diff)  # Higher reward for closer to target height
    
    # Reward for keeping the cube static
    if check_actor_static(self.obj):
        reward_static = 1.0  # Full reward if the cube is static
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static
    )
    
    return reward