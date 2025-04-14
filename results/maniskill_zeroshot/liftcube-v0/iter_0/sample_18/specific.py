import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.5
    weight_static = 0.2
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0   # Reward for lifting the cube
    reward_static = 0.0 # Reward for keeping the cube static during lifting
    
    # Get positions
    tcp_pos = self.tcp.pose.p  # TCP position
    cube_pos = self.obj.pose.p     # Cube position
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for approaching and grasping the cube
    if distance < 0.02 * 2:  # Close enough to grasp
        if self.agent.check_grasp(self.obj):  # Successful grasp
            reward_grasp = 1.0
    
    # Reward for lifting the cube
    if reward_grasp > 0.0:  # Only if the cube is grasped
        target_height = 0.2  # Target lift height
        current_height = cube_pos[2]  # Current height of the cube
        reward_lift = max(0.0, 1.0 - np.abs(target_height - current_height) / target_height)
    
    # Reward for keeping the cube static during lifting
    if reward_grasp > 0.0 and reward_lift > 0.0:  # Only if the cube is grasped and lifted
        if check_actor_static(self.obj):  # Cube is static
            reward_static = 1.0
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static
    )
    
    return reward