import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_static = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping cube A
    reward_lift = 0.0   # Reward for lifting cube A by 0.2 meters
    reward_static = 0.0 # Reward for keeping cube A static during lifting
    reward_control = 0.0 # Reward for minimizing control effort
    
    # Calculate grasp reward
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate lift reward
    cube_height = self.obj.pose.p[2] - 0.02
    target_height = 0.2
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0.0, 1.0 - height_diff / target_height)
    
    # Calculate static reward
    if check_actor_static(self.obj):
        reward_static = 1.0
    
    # Calculate control reward
    control_effort = np.linalg.norm(self.agent.robot.get_qvel()[:-2]) + np.linalg.norm(action)
    reward_control = max(0.0, 1.0 - control_effort / 10.0)  # Normalize control effort
    
    # Combine all rewards
    reward = (weight_grasp * reward_grasp +
              weight_lift * reward_lift +
              weight_static * reward_static +
              weight_control * reward_control)
    
    return reward