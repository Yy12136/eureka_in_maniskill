import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0  # Reward for lifting the cube by 0.2 meters
    reward_static = 0.0  # Reward for keeping the cube static after lifting
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Grasping reward
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Reward for successful grasp
        
    # Lifting reward
    target_height = 0.2  # Target height to lift the cube
    current_height = cube_pos[2]  # Current height of the cube
    if self.agent.check_grasp(self.obj):
        reward_lift = 1.0 - np.tanh(abs(current_height - target_height))  # Reward for lifting to target height
        
    # Static reward
    if check_actor_static(self.obj) and self.agent.check_grasp(self.obj):
        reward_static = 1.0  # Reward for keeping the cube static after lifting
        
    # Control effort reward
    qvel = self.agent.robot.get_qvel()[:-2]
    reward_control = -np.tanh(np.linalg.norm(qvel))  # Penalize high velocity to encourage smooth control
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward