import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.45126891882483305
    weight_lift = 0.4487310811751659
    weight_static = 0.05000000000000043
    weight_control = 0.05000000000000053
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0   # Reward for lifting the cube
    reward_static = 0.0 # Reward for keeping the cube static after lifting
    reward_control = 0.0 # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - np.tanh(distance)  # Higher reward when closer to the cube
    
    # Reward for lifting the cube by 0.2 meters
    target_height = 0.2
    current_height = cube_pos[2] - 0.02
    if current_height >= target_height:
        reward_lift = 1.0
    else:
        reward_lift = np.tanh(current_height / target_height)  # Gradually increase reward as height increases
    
    # Reward for keeping the cube static after lifting
    if current_height >= target_height and check_actor_static(self.obj):
        reward_static = 1.0
    
    # Reward for minimizing control effort
    control_effort = np.linalg.norm(self.agent.robot.get_qvel()[:-2]) + np.linalg.norm(action)
    reward_control = 1.0 / (1.0 + control_effort)  # Higher reward for lower control effort
    
    # Combine all rewards
    reward = (weight_grasp * reward_grasp +
              weight_lift * reward_lift +
              weight_static * reward_static +
              weight_control * reward_control)
    
    return reward