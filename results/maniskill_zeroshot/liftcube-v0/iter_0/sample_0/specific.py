import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_stability = 0.2
    weight_control = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping cube A
    reward_lift = 0.0   # Reward for lifting cube A by 0.2 meters
    reward_stability = 0.0  # Reward for maintaining stability during the task
    reward_control = 0.0  # Reward for smooth and controlled robot movements
    
    # Calculate reward for grasping cube A
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate reward for lifting cube A by 0.2 meters
    cube_height = self.obj.pose.p[2] - 0.02
    target_height = 0.2
    height_diff = max(0, target_height - cube_height)
    reward_lift = 1.0 - (height_diff / target_height)
    
    # Calculate reward for stability
    if check_actor_static(self.obj):
        reward_stability = 1.0
    
    # Calculate reward for smooth and controlled robot movements
    qvel_norm = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward_control = 1.0 / (1.0 + qvel_norm)
    
    # Combine all rewards
    reward = (weight_grasp * reward_grasp +
              weight_lift * reward_lift +
              weight_stability * reward_stability +
              weight_control * reward_control)
    
    return reward