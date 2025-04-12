import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for approaching the cube
    if distance < 0.1:
        reward_grasp = 1.0 - np.tanh(10.0 * distance)
    
    # Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
        
        # Reward for lifting the cube
        target_height = 0.2
        current_height = cube_pos[2] - 0.02
        height_diff = target_height - current_height
        reward_lift = 1.0 - np.tanh(10.0 * height_diff)
        
        # Reward for keeping the cube static
        if check_actor_static(self.obj):
            reward_static = 1.0
    
    # Reward for minimizing control effort
    qvel = self.agent.robot.get_qvel()[:-2]
    reward_control = 1.0 - np.tanh(0.1 * np.linalg.norm(qvel))
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward