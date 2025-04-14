import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for getting close to the cube
    if distance < 0.02:
        reward_grasp = 1.0 - distance / 0.02
    
    # Reward for successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube
    target_height = 0.2
    current_height = cube_pos[2]
    if self.agent.check_grasp(self.obj):
        reward_lift = 1.0 - abs(current_height - target_height) / target_height
    
    # Reward for keeping the cube static
    if check_actor_static(self.obj):
        reward_static = 1.0
    
    # Reward for minimizing control effort
    qvel = self.agent.robot.get_qvel()[:-2]
    reward_control = 1.0 - np.linalg.norm(qvel) / 10.0
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward