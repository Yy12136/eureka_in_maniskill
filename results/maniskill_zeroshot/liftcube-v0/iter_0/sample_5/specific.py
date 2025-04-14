import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0  # Reward for lifting the cube by 0.2 meters
    reward_static = 0.0  # Reward for keeping the cube static after lifting
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for approaching the cube
    if distance < 0.02 * 2:
        reward_grasp = 1.0 - min(distance / (0.02 * 2), 1.0)
    
    # Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
        
        # Calculate the height difference
        initial_height = self.obj.pose.p[2]  # Initial z position of the cube
        current_height = cube_pos[2]
        height_diff = current_height - initial_height
        
        # Reward for lifting the cube by 0.2 meters
        if height_diff >= 0.2:
            reward_lift = 1.0
        else:
            reward_lift = min(height_diff / 0.2, 1.0)
        
        # Reward for keeping the cube static after lifting
        if check_actor_static(self.obj):
            reward_static = 1.0
    
    # Reward for minimizing control effort
    control_effort = np.linalg.norm(self.agent.robot.get_qvel()[:-2]) + np.linalg.norm(action)
    reward_control = 1.0 / (1.0 + control_effort)
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward