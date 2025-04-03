import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get chair and target positions
    chair_pos = self.root_link.pose.p[:2]  # XY position of the chair
    target_pos = self.target_xy  # XY position of the target
    
    # Calculate distance between chair and target
    distance_to_target = np.linalg.norm(chair_pos - target_pos)
    
    # Stage 1: Approach the Chair
    # Encourage the robot to move its base and arms close to the chair
    ee_coords = self.agent.get_ee_coords()  # Get end-effector coordinates
    chair_pcd = self.env.env._get_chair_pcd()  # Get chair point cloud
    min_dist_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min()
    approach_reward = -min_dist_to_chair  # Closer to the chair is better
    reward += approach_reward * 0.3
    
    # Stage 2: Grasp the Chair
    # Encourage the robot to position its grippers appropriately to push the chair
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])  # Calculate tilt of the chair
    grasp_reward = -chair_tilt  # Less tilt is better
    reward += grasp_reward * 0.2
    
    # Stage 3: Push the Chair
    # Encourage the robot to move the chair towards the target
    push_reward = -distance_to_target  # Closer to the target is better
    reward += push_reward * 0.3
    
    # Stage 4: Stabilize the Chair
    # Penalize the robot if the chair tilts too much
    tilt_penalty = max(0, chair_tilt - np.pi/6)  # Penalize if tilt > 30 degrees
    reward -= tilt_penalty * 0.2
    
    # Stage 5: Reach the Target
    # Provide a large reward if the chair reaches the target
    if distance_to_target < 0.1:  # Threshold for reaching the target
        target_reward = 1.0
        reward += target_reward
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smooth movements
    action_penalty = np.linalg.norm(action)
    reward -= action_penalty * 0.1
    
    return reward