import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Reach the cube
    distance_to_cube = np.linalg.norm(ee_pos - cubeA_pos)
    reach_reward = -distance_to_cube  # Penalize distance to cube
    reward += reach_reward * 0.5
    
    # Milestone 2: Grasp the cube
    if grasp_success:
        reward += 1.0  # Reward for successful grasp
    else:
        reward -= 0.5  # Penalize for not grasping
    
    # Milestone 3: Lift the cube by 0.2 meters
    if grasp_success:
        target_height = cubeA_pos[2] + 0.2
        height_diff = abs(cubeA_pos[2] - target_height)
        lift_reward = -height_diff  # Penalize deviation from target height
        reward += lift_reward * 1.0
    
    # Milestone 4: Ensure the cube is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 0.5  # Reward for keeping the cube static
    
    # Penalize large actions to encourage smooth movements
    action_penalty = -np.linalg.norm(action) * 0.01
    reward += action_penalty
    
    return reward