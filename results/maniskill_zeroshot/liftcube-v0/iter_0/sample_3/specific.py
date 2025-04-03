import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_stability = 0.2
    weight_effort = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping cube A
    reward_lift = 0.0   # Reward for lifting cube A by 0.2 meters
    reward_stability = 0.0  # Reward for maintaining stability during the task
    reward_effort = 0.0  # Reward for minimizing effort (joint velocity)
    
    # Calculate reward for grasping
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate reward for lifting
    ee_pose = self.tcp.pose
    cube_height = ee_pose[2] - 0.02
    if cube_height >= 0.2:
        reward_lift = 1.0
    else:
        reward_lift = cube_height / 0.2  # Linear scaling based on height
    
    # Calculate reward for stability
    if check_actor_static(self.obj):
        reward_stability = 1.0
    
    # Calculate reward for effort (minimize joint velocity)
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward_effort = 1.0 - (np.linalg.norm(joint_velocities) / 10.0)  # Normalize by max expected velocity
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_stability * reward_stability +
        weight_effort * reward_effort
    )
    
    return reward