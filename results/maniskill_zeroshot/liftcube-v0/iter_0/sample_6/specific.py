import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasping
    reward_lift = 0.0   # Reward for lifting the cube
    reward_static = 0.0 # Reward for keeping the cube static
    reward_control = 0.0 # Reward for smooth control
    
    # Get positions and states
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    cube_static = check_actor_static(self.obj)
    is_grasped = self.agent.check_grasp(self.obj)
    
    # Reward for successful grasping
    if is_grasped:
        reward_grasp = 1.0
    
    # Reward for lifting the cube by 0.2 meters
    target_height = 0.2
    current_height = cube_pos[2] - 0.02
    height_diff = max(0, target_height - current_height)
    reward_lift = 1.0 - (height_diff / target_height)
    
    # Reward for keeping the cube static
    if cube_static:
        reward_static = 1.0
    
    # Reward for smooth control (minimize velocity and action magnitude)
    qvel = self.agent.robot.get_qvel()[:-2]
    action_magnitude = np.linalg.norm(action)
    reward_control = 1.0 / (1.0 + np.linalg.norm(qvel) + action_magnitude)
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward