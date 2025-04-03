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
        dist_to_cube = np.linalg.norm(ee_pos - obj_pos)
        reach_reward = 1.0 / (1.0 + dist_to_cube)
        reward += reach_reward
    else:
        # Step 2: Lift the cube
        # Target height is 0.2 meters above the initial position
        target_height = obj_pos[2] + 0.2
        current_height = ee_pos[2]
        
        # Reward for lifting the cube closer to the target height
        height_diff = abs(target_height - current_height)
        lift_reward = 1.0 / (1.0 + height_diff)
        reward += lift_reward
        
        # Additional reward if the cube is static (not wobbling)
        if check_actor_static(self.obj):
            reward += 0.5
    
    # Step 3: Penalize excessive joint velocities
    joint_vel = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    vel_penalty = -0.01 * joint_vel
    reward += vel_penalty
    
    # Step 4: Penalize large joint position changes
    joint_pos_diff = np.linalg.norm(self.agent.robot.get_qpos()[:-2] - self.agent.robot.get_qpos()[:-2]_prev)
    pos_penalty = -0.01 * joint_pos_diff
    reward += pos_penalty
    
    return reward