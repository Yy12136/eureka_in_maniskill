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
    reward_effort = 0.0  # Reward for minimizing effort (joint velocities)
    
    # Calculate reward_grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate reward_lift
    cube_height = self.obj.pose.p[2]  # Assuming z-coordinate represents height
    target_height = 0.2
    height_diff = abs(cube_height - target_height)
    reward_lift = max(0, 1 - height_diff / target_height)
    
    # Calculate reward_stability
    if check_actor_static(self.obj):
        reward_stability = 1.0
    
    # Calculate reward_effort
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    effort = np.linalg.norm(joint_velocities)
    reward_effort = max(0, 1 - effort / 10.0)  # Normalize effort to a reasonable range
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_lift * reward_lift +
        weight_stability * reward_stability +
        weight_effort * reward_effort
    )
    
    return reward