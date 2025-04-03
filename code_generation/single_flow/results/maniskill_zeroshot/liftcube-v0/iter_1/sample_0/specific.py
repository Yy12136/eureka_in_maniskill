import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Stage 1: Move the gripper close to cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
    
    # Use exponential decay for distance reward to encourage smooth approach
    reward += np.exp(-5 * distance_to_cubeA)  # Weight: 1.0
    
    # Stage 2: Grasp cube A
    if distance_to_cubeA < 0.05:  # If gripper is close enough to cube A
        # Check if cube A is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        
        # Reward for successful grasp
        if is_grasped:
            reward += 1.0  # Weight: 1.0
        else:
            # Penalize for not grasping when close enough
            reward -= 0.5  # Weight: -0.5
    
    # Stage 3: Lift cube A by 0.2 meters
    if self.agent.check_grasp(self.obj, max_angle=30):  # If cube A is grasped
        # Calculate the height difference between cube A and the goal height
        current_height = self.obj.pose.p[2]  # Z-axis position
        height_difference = abs(current_height - self.goal_height)
        
        # Reward for lifting cube A towards the goal height
        reward += np.exp(-5 * height_difference)  # Weight: 1.0
        
        # Penalize if the cube is dropped or moved away from the goal height
        if height_difference > 0.2:
            reward -= 0.5  # Weight: -0.5
    
    # Regularization of the robot's action
    # Penalize large actions and abrupt changes to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)  # Weight: -0.01
    action_smoothness_penalty = -0.01 * np.linalg.norm(np.diff(action))  # Weight: -0.01
    reward += action_penalty + action_smoothness_penalty
    
    return reward