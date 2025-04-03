import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Stage 1: Move the gripper close to cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
    
    # Reward for reducing the distance to cube A
    reward += max(0, 1 - distance_to_cubeA / 0.1)  # Normalize distance to [0, 1]
    
    # Stage 2: Grasp cube A
    if distance_to_cubeA < 0.05:  # If gripper is close enough to cube A
        # Check if cube A is grasped
        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        
        # Reward for successful grasp
        if is_grasped:
            reward += 1.0
        else:
            # Reward for closing the gripper around the cube, even if not fully grasped
            gripper_openness = self.robot.gripper_openness
            reward += max(0, 1 - gripper_openness)  # Reward for closing the gripper
    
    # Stage 3: Lift cube A by 0.2 meters
    if self.agent.check_grasp(self.obj, max_angle=30):  # If cube A is grasped
        # Calculate the height difference between cube A and the goal height
        current_height = self.obj.pose.p[2]  # Z-axis position
        height_difference = abs(current_height - self.goal_height)
        
        # Reward for lifting cube A towards the goal height
        reward += max(0, 1 - height_difference / 0.2)  # Normalize height difference to [0, 1]
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    # Penalize high velocity to encourage stable movements
    joint_velocity = np.linalg.norm(self.agent.robot.get_qvel()[:-2])
    velocity_penalty = -0.01 * joint_velocity
    reward += velocity_penalty
    
    return reward