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
    reward += max(0, 1.0 - distance_to_cube) * 0.3  # Reduced weight for smoother guidance
    
    # Milestone 2: Grasp cube A
    if grasp_success:
        reward += 1.0  # Significant reward for successful grasp
    
    # Milestone 3: Lift cube A by 0.2 meter
    if grasp_success:
        target_height = obj_pos[2] + 0.2
        current_height = obj_pos[2]
        height_diff = abs(target_height - current_height)
        reward += max(0, 1.0 - height_diff) * 0.5  # Reward based on height progress
    
    # Milestone 4: Ensure cube A is static after lifting
    if grasp_success and check_actor_static(self.obj):
        reward += 1.0  # Reward for stability
    
    # Penalize large actions to encourage smooth motion
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    # Penalize collisions or unstable states
    if not grasp_success and distance_to_cube < 0.02:
        reward -= 0.5  # Penalize accidental collisions
    
    # Encourage efficient motion by penalizing unnecessary joint movements
    joint_velocity_penalty = -0.001 * np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    reward += joint_velocity_penalty
    
    return reward