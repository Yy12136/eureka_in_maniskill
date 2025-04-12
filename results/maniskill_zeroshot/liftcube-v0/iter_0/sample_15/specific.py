import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.14999999999987806
    weight_lift = 0.7999999999987442
    weight_static = 0.05000000000137785
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static during lifting
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for approaching the cube
    if not self.agent.check_grasp(self.obj):
        reward_grasp = max(0, 1 - distance / 0.1)  # Normalized reward based on distance
    
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube
    if self.agent.check_grasp(self.obj):
        target_height = cube_pos[2] + 0.2  # Target height is 0.2 meters above current position
        current_height = cube_pos[2]
        reward_lift = max(0, 1 - abs(target_height - current_height) / 0.2)  # Normalized reward based on height
    
    # Reward for keeping the cube static during lifting
    if self.agent.check_grasp(self.obj) and check_actor_static(self.obj):
        reward_static = 1.0
    
    # Combine all rewards
    reward = weight_grasp * reward_grasp + weight_lift * reward_lift + weight_static * reward_static
    
    return reward