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
    reward += max(0, 1.0 - distance_to_cube) * 0.3  # Reduced weight to prioritize grasping
    
    # Milestone 2: Align end-effector with cube A (orientation matters)
    ee_z_axis = self.tcp.pose.to_transformation_matrix()[:3, 2]
    desired_z_axis = np.array([0, 0, -1])  # Assuming downward grasp
    alignment = np.dot(ee_z_axis, desired_z_axis)
    reward += max(0, alignment) * 0.2  # Encourage proper alignment
    
    # Milestone 3: Grasp cube A
    if grasp_success:
        reward += 1.0  # Significant reward for successful grasp
    
    # Milestone 4: Lift cube A by 0.2 meter
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        current_height = obj_pos[2]
        height_diff = target_height - current_height
        reward += max(0, 1.0 - height_diff) * 0.5  # Encourage lifting to target height
    
    # Milestone 5: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Reward for stability after lifting
    
    # Penalize large actions to encourage smooth motion
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    # Penalize excessive robot velocity (encourage smooth motion)
    qvel = self.agent.robot.get_qvel()[:-2]
    velocity_penalty = -0.005 * np.linalg.norm(qvel)
    reward += velocity_penalty
    
    return reward