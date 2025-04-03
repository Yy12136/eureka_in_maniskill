import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Reach the cube
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reach_reward = 1.0 - np.tanh(5.0 * distance_to_cube)
    reward += reach_reward
    
    # Milestone 2: Grasp the cube
    if grasp_success:
        reward += 1.0
    
    # Milestone 3: Lift the cube by 0.2 meter
    if grasp_success:
        target_height = 0.2
        current_height = obj_pos[2] - 0.02
        height_diff = target_height - current_height
        lift_reward = 1.0 - np.tanh(10.0 * max(0.0, height_diff))
        reward += lift_reward
    
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    # Penalize high velocities to encourage stability
    qvel_penalty = -0.01 * np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward += qvel_penalty
    
    return reward