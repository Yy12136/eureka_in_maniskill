import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Move end-effector close to cube A
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    approach_reward = max(0, 1.0 - distance_to_cube / 0.1)  # Normalize distance to 0.1m
    reward += approach_reward * 0.3
    
    # Milestone 2: Grasp cube A
    if grasp_success:
        reward += 1.0
    
    # Milestone 3: Lift cube A by 0.2 meter
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        current_height = obj_pos[2]
        height_diff = max(0, target_height - current_height)
        height_progress = height_diff / 0.2  # Normalize progress to 0.2m
        reward += height_progress * 0.5
    
    # Milestone 4: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0
    
    # Penalize large actions to encourage smooth motion
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward