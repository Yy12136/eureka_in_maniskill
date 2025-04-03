import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Stage 1: Move the gripper close to cube A
    # Calculate the distance between the gripper and cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
    
    # Reward for reducing the distance to cube A
    # Use a smooth exponential decay for distance reward
    reward += np.exp(-5 * distance_to_cubeA)  # More continuous and smooth reward
    
    # Stage 2: Grasp cube A
    if distance_to_cubeA < 0.05:  # If gripper is close enough to cube A
        # Check if cube A is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        
        # Reward for successful grasp
        if is_grasped:
            reward += 1.0
        else:
            # Continuous reward based on gripper openness and proximity
            gripper_openness = self.robot.gripper_openness
            # Encourage gripper to close when close to the cube
            reward += 0.5 * (1 - gripper_openness) * np.exp(-5 * distance_to_cubeA)
    
    # Stage 3: Lift cube A by 0.2 meters
    if self.agent.check_grasp(self.obj, max_angle=30):  # If cube A is grasped
        # Calculate the height difference between cube A and the goal height
        current_height = self.obj.pose.p[2]  # Z-axis position
        height_difference = abs(current_height - self.goal_height)
        
        # Reward for lifting cube A towards the goal height
        # Use a smooth exponential decay for height difference
        reward += np.exp(-5 * height_difference)
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward