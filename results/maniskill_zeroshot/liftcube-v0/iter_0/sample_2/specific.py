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
    reward_static = 0.0 # Reward for ensuring cube A is static after lifting
    reward_control = 0.0 # Reward for minimizing control effort
    
    # Calculate each component
    
    # Grasp reward: Check if cube A is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Lift reward: Check if cube A is lifted by 0.2 meters
    ee_z = self.tcp.pose.p[2]
    cube_z = self.obj.pose.p[2]
    lift_height = cube_z - 0.02
    if lift_height >= 0.2:
        reward_lift = 1.0
    else:
        reward_lift = lift_height / 0.2  # Proportional reward based on height
    
    # Static reward: Check if cube A is static after lifting
    if check_actor_static(self.obj):
        reward_static = 1.0
    
    # Control reward: Minimize control effort (joint velocities)
    qvel_norm = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward_control = 1.0 / (1.0 + qvel_norm)  # Inverse proportional to joint velocities
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_static * reward_static +
        weight_control * reward_control
    )
    
    return reward