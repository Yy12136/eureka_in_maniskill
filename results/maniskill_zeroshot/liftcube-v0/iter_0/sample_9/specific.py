import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Step 1: Reach the cube
    if not grasp_success:
        # Reward for moving the end-effector closer to the cube
        distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
        reward += 1.0 / (1.0 + distance_to_cube)
        
        # Penalize large velocities to encourage smooth motion
        qvel_norm = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
        reward -= 0.01 * qvel_norm
    else:
        # Step 2: Lift the cube by 0.2 meters
        target_height = obj_pos[2] + 0.2
        current_height = ee_pos[2]
        
        # Reward for lifting the cube towards the target height
        height_diff = abs(current_height - target_height)
        reward += 1.0 / (1.0 + height_diff)
        
        # Penalize if the cube is not static (e.g., shaking)
        if not check_actor_static(self.obj):
            reward -= 0.1
        
        # Penalize large velocities to encourage smooth lifting
        qvel_norm = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
        reward -= 0.01 * qvel_norm
    
    # Step 3: Final milestone - cube is lifted to the target height
    if grasp_success and abs(ee_pos[2] - (obj_pos[2] + 0.2)) < 0.01 and check_actor_static(self.obj):
        reward += 10.0  # Large reward for successfully completing the task
    
    return reward