import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Step 1: Approach Cube A
    # Calculate the distance between the gripper and cube A
    gripper_pos = self.tcp.pose.p
    cubeA_pos = self.obj.pose.p
    distance_to_cubeA = np.linalg.norm(gripper_pos - cubeA_pos)
    
    # Reward for reducing the distance to cube A
    reward += 0.5 * (1 - np.tanh(10 * distance_to_cubeA))
    
    # Step 2: Grasp Cube A
    if distance_to_cubeA < 0.05:  # Threshold for being close enough to grasp
        # Check if the gripper is closed enough to grasp cube A
        if self.robot.gripper_openness < 0.1:  # Threshold for gripper closedness
            # Check if the gripper is actually grasping cube A
            if self.agent.check_grasp(self.obj, max_angle=30):
                reward += 0.3  # Reward for successful grasp
                
                # Step 3: Lift Cube A
                # Calculate the height difference between cube A and the target height
                current_height = self.obj.pose.p[2]
                target_height = self.obj.pose.p[2] + self.goal_height
                height_difference = abs(current_height - target_height)
                
                # Reward for lifting cube A closer to the target height
                reward += 0.2 * (1 - np.tanh(10 * height_difference))
                
                # Additional reward if cube A is lifted to the target height
                if height_difference < 0.01:  # Threshold for reaching the target height
                    reward += 0.5  # Reward for successfully lifting cube A
    
    # Step 4: Regularization
    # Penalize large actions to ensure smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty
    
    return reward