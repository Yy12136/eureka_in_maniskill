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
    # Use a more aggressive reward scaling for closer distances
    reward += max(0, 1 - distance_to_cubeA / 0.05) * 0.5  # Weight: 0.5
    
    # Stage 2: Align the gripper with cube A
    # Calculate the angle between the gripper and cube A
    gripper_orientation = self.tcp.pose.q
    cubeA_orientation = self.obj.pose.q
    angle_diff = np.arccos(2 * np.square(np.dot(gripper_orientation, cubeA_orientation)) - 1)
    
    # Reward for aligning the gripper with cube A
    reward += max(0, 1 - angle_diff / np.pi) * 0.3  # Weight: 0.3
    
    # Stage 3: Grasp cube A
    if distance_to_cubeA < 0.03:  # If gripper is close enough to cube A
        # Check if cube A is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        
        # Reward for successful grasp
        if is_grasped:
            reward += 1.0  # Weight: 1.0
        else:
            # Penalize for not grasping when close enough
            reward -= 0.5  # Penalty: -0.5
    
    # Stage 4: Lift cube A by 0.2 meters
    if self.agent.check_grasp(self.obj, max_angle=30):  # If cube A is grasped
        # Calculate the height difference between cube A and the goal height
        current_height = self.obj.pose.p[2]  # Z-axis position
        height_difference = abs(current_height - self.goal_height)
        
        # Reward for lifting cube A towards the goal height
        reward += max(0, 1 - height_difference / 0.2) * 0.8  # Weight: 0.8
        
        # Penalize high velocities to prevent jerky movements
        cubeA_velocity = np.linalg.norm(self.obj.velocity)
        reward -= min(cubeA_velocity, 1.0) * 0.1  # Penalty: -0.1
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward