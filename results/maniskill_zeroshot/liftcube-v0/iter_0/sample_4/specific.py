import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_approach = 0.3  # Reward for approaching the cube
    weight_grasp = 0.3     # Reward for successfully grasping the cube
    weight_lift = 0.4      # Reward for lifting the cube to the desired height
    
    # Initialize components
    reward_approach = 0.0  # Component for approaching the cube
    reward_grasp = 0.0     # Component for grasping the cube
    reward_lift = 0.0      # Component for lifting the cube
    
    # Calculate the distance between the end-effector and the cube
    ee_pos = self.tcp.pose[:3]
    cube_pos = self.obj.pose[:3]
    distance = np.linalg.norm(ee_pos - cube_pos)
    
    # Reward for approaching the cube
    reward_approach = max(0.0, 1.0 - distance / (0.02 * 2))
    
    # Reward for successfully grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Reward for lifting the cube to the desired height
    if self.agent.check_grasp(self.obj):
        lifted_height = self.obj.pose[2] - cube_pos[2]
        reward_lift = max(0.0, 1.0 - abs(lifted_height - 0.2) / 0.2)
    
    # Combine all rewards
    reward = weight_approach * reward_approach + weight_grasp * reward_grasp + weight_lift * reward_lift
    
    # Consider adding any necessary regularization or additional terms
    # For example, penalize large joint velocities or positions
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    joint_position_penalty = -0.01 * np.linalg.norm(self.agent.robot.get_qpos()[:-2])
    
    reward += joint_velocity_penalty + joint_position_penalty
    
    return reward