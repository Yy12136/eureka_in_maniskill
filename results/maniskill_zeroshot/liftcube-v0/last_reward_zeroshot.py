import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping the cube
    reward_lift = 0.0  # Reward for lifting the cube to the desired height
    reward_static = 0.0  # Reward for keeping the cube static after lifting
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Grasp reward: Encourage the robot to grasp the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Higher reward for closer grasp
    
    # Lift reward: Encourage the robot to lift the cube by 0.2 meters
    target_height = 0.2
    current_height = cube_pos[2]  # Z-coordinate of the cube
    height_diff = abs(current_height - target_height)
    reward_lift = 1.0 - np.tanh(height_diff)  # Higher reward for closer to target height
    
    # Static reward: Encourage the cube to remain static after lifting
    if check_actor_static(self.obj):
        reward_static = 1.0  # Full reward if the cube is static
    
    # Control reward: Encourage minimal control effort
    control_effort = np.linalg.norm(action)
    reward_control = 1.0 - np.tanh(control_effort)  # Higher reward for less control effort
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward