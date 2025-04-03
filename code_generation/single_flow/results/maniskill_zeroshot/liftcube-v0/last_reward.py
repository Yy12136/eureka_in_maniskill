import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get the current pose of cube A
    cubeA_pos = self.cubeA.pose.p
    cubeA_quat = self.cubeA.pose.q
    
    # Get the current pose of the robot's end-effector
    ee_pos = self.robot.ee_pose.p
    ee_quat = self.robot.ee_pose.q
    
    # Get the current openness of the gripper
    gripper_openness = self.robot.gripper_openness
    
    # Stage 1: Approach Cube A
    # Reward for reducing the distance between the end-effector and cube A
    distance_to_cubeA = np.linalg.norm(ee_pos - cubeA_pos)
    approach_reward = -distance_to_cubeA  # Negative reward to minimize distance
    reward += approach_reward * 0.5  # Weight for approach stage
    
    # Stage 2: Grasp Cube A
    if distance_to_cubeA < 0.05:  # If the end-effector is close enough to cube A
        # Reward for closing the gripper
        grasp_reward = -gripper_openness  # Negative reward to minimize openness
        reward += grasp_reward * 0.3  # Weight for grasp stage
        
        # Check if the gripper is successfully grasping cube A
        if self.robot.check_grasp(self.cubeA, max_angle=30):
            # Stage 3: Lift Cube A
            # Reward for lifting cube A by 0.2 meters
            target_height = self.goal_height
            current_height = cubeA_pos[2]  # Z-axis position
            height_difference = current_height - target_height
            lift_reward = -np.abs(height_difference)  # Negative reward to minimize height difference
            reward += lift_reward * 0.2  # Weight for lift stage
            
            # Additional reward for maintaining the grasp during lifting
            if height_difference > 0:  # If cube A is being lifted
                grasp_maintenance_reward = -gripper_openness  # Encourage maintaining the grasp
                reward += grasp_maintenance_reward * 0.1  # Weight for grasp maintenance
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -np.linalg.norm(action) * 0.1
    reward += action_penalty
    
    return reward