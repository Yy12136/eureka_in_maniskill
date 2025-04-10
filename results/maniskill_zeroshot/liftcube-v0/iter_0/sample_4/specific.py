import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.45126891882483305
    weight_lift = 0.4487310811751659
    weight_static = 0.05000000000000043
    weight_control = 0.05000000000000053
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0  # Reward for lifting the cube
    reward_static = 0.0  # Reward for keeping the cube static
    reward_control = 0.0  # Reward for minimizing control effort
    
    # Get positions and states
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    cube_static = check_actor_static(self.obj)
    is_grasped = self.agent.check_grasp(self.obj)
    
    # Reward for successful grasp
    if is_grasped:
        reward_grasp = 1.0
    
    # Reward for lifting the cube by 0.2 meters
    target_height = 0.2
    current_height = cube_pos[2] - 0.02
    height_diff = target_height - current_height
    if is_grasped:
        reward_lift = max(0.0, 1.0 - abs(height_diff) / target_height)
    
    # Reward for keeping the cube static
    if cube_static:
        reward_static = 1.0
    
    # Reward for minimizing control effort
    qpos = self.agent.robot.get_qpos()[:-2]
    qvel = self.agent.robot.get_qvel()[:-2]
    control_effort = np.linalg.norm(qpos) + np.linalg.norm(qvel)
    reward_control = max(0.0, 1.0 - control_effort / 10.0)  # Normalize control effort
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward