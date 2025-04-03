import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_approach = 0.3
    weight_grasp = 0.3
    weight_lift = 0.4
    
    # Initialize components
    reward_approach = 0.0  # Reward for approaching the cube
    reward_grasp = 0.0     # Reward for successfully grasping the cube
    reward_lift = 0.0      # Reward for lifting the cube by 0.2 meters
    
    # Calculate reward_approach: Encourage the robot to move closer to the cube
    ee_pos = self.tcp.pose[:3]
    cube_pos = self.obj.pose[:3]
    distance_to_cube = np.linalg.norm(ee_pos - cube_pos)
    reward_approach = max(0, 1 - distance_to_cube / (0.02 * 2))
    
    # Calculate reward_grasp: Reward if the cube is successfully grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # Calculate reward_lift: Reward for lifting the cube by 0.2 meters
    if self.agent.check_grasp(self.obj):
        cube_height = self.obj.pose[2]
        initial_height = cube_pos[2]
        height_diff = cube_height - initial_height
        reward_lift = max(0, min(height_diff / 0.2, 1))
    
    # Combine all rewards
    reward = (
        weight_approach * reward_approach +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    return reward