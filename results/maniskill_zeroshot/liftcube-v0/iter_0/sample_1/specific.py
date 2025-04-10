import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.5
    weight_static = 0.2
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping the cube
    reward_lift = 0.0   # Reward for lifting the cube to the target height
    reward_static = 0.0 # Reward for keeping the cube static during lifting
    
    # Get positions
    tcp_pos = self.tcp.pose.p  # TCP (Tool Center Point) position
    cube_pos = self.obj.pose.p     # Cube A position
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Grasping reward
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Reward increases as distance decreases
    
    # Lifting reward
    target_height = 0.2  # Target height to lift the cube
    current_height = cube_pos[2] - 0.02  # Current height of the cube
    if self.agent.check_grasp(self.obj):
        reward_lift = 1.0 - np.tanh(abs(target_height - current_height))  # Reward increases as height approaches target
    
    # Static reward
    if self.agent.check_grasp(self.obj) and check_actor_static(self.obj):
        reward_static = 1.0  # Reward for keeping the cube static during lifting
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static
    )
    
    return reward