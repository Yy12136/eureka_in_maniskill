import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3613070476824337
    weight_lift = 0.27738590463515
    weight_static = 0.3613070476824163
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping the cube
    reward_lift = 0.0   # Reward for lifting the cube by 0.2 meters
    reward_static = 0.0 # Reward for keeping the cube static during lifting
    
    # Get positions
    tcp_pos = self.tcp.pose.p  # TCP (Tool Center Point) position
    cube_pos = self.obj.pose.p    # Cube A position
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Higher reward for closer grasp
    
    # Reward for lifting the cube by 0.2 meters
    if self.agent.check_grasp(self.obj):
        target_height = 0.2
        current_height = cube_pos[2] - 0.02  # Subtract half size to get base height
        reward_lift = 1.0 - np.tanh(abs(target_height - current_height))  # Higher reward for closer to target height
    
    # Reward for keeping the cube static during lifting
    if self.agent.check_grasp(self.obj) and check_actor_static(self.obj):
        reward_static = 1.0  # Full reward if the cube is static
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static
    )
    
    return reward