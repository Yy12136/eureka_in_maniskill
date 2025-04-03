import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions of gripper and cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    
    # Stage 1: Move the gripper close to cube A
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
    reward += max(0, 1 - distance_to_cubeA / 0.1)  # Normalize distance to [0, 1]
    
    # Stage 2: Grasp cube A
    if distance_to_cubeA < 0.05:  # If gripper is close enough to cube A
        # Check if cube A is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        
        # Reward for successful grasp
        if is_grasped:
            reward += 1.0
            # Additional reward for grasp quality (centered grasp)
            grasp_center = (self.robot.lfinger.pose.p + self.robot.rfinger.pose.p) / 2
            grasp_quality = 1 - np.linalg.norm(grasp_center - cubeA_pos) / 0.02
            reward += 0.5 * grasp_quality
        else:
            # Penalize for not grasping when close enough
            reward -= 0.5
    
    # Stage 3: Lift cube A by 0.2 meters
    if self.agent.check_grasp(self.obj, max_angle=30):  # If cube A is grasped
        # Calculate the height difference between cube A and the goal height
        initial_height = self.obj.pose.p[2]  # Initial Z-axis position
        current_height = cubeA_pos[2]
        height_difference = abs(current_height - (initial_height + self.goal_height))
        
        # Reward for lifting cube A towards the goal height
        reward += max(0, 1 - height_difference / 0.2)  # Normalize height difference to [0, 1]
        
        # Penalize for instability (cube A should not tilt or drop)
        cubeA_velocity = np.linalg.norm(self.obj.velocity)
        reward -= 0.1 * cubeA_velocity  # Penalize for high velocity
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.05 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward