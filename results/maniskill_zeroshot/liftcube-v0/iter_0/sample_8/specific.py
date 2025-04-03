import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Approach the cube
    # Reward for reducing the distance between the end-effector and the cube
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reward += 1.0 / (1.0 + distance_to_cube)  # Inverse distance reward
    
    # Milestone 2: Grasp the cube
    if grasp_success:
        reward += 1.0  # Bonus for successful grasp
    
    # Milestone 3: Lift the cube by 0.2 meter
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        height_difference = abs(obj_pos[2] - target_height)
        reward += 1.0 / (1.0 + height_difference)  # Inverse height difference reward
    
    # Milestone 4: Keep the cube static while lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 0.5  # Bonus for keeping the cube static
    
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward