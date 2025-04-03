import numpy as np
from scipy.spatial.distance import cdist

class BaseEnv(gym.Env):
    def compute_dense_reward(self, action) -> float:
        # Initialize reward
        reward = 0.0

        # Get the chair's current position and target position
        chair_position = self.chair.pose.p[:2]  # Only consider x, y coordinates
        target_position = self.target_xy

        # 1. Distance to Target
        distance_to_target = np.linalg.norm(chair_position - target_position)
        reward -= 0.5 * distance_to_target  # Penalize distance to target

        # 2. Chair Stability
        z_axis_chair = self.chair.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])  # Angle between chair's z-axis and world z-axis
        reward -= 0.3 * chair_tilt  # Penalize chair tilt

        # 3. Robot Action Regularization
        action_penalty = 0.01 * np.linalg.norm(action)
        reward -= action_penalty  # Penalize large actions

        # 4. Staged Rewards
        # Stage 1: Approach the chair
        if distance_to_target > 0.5:  # If the chair is far from the target
            # Encourage the robot to move towards the chair
            ee_coords = self.robot.get_ee_coords()
            chair_pcd = self.chair.get_pcd()
            min_distance_to_chair = cdist(ee_coords.reshape(-1, 3), chair_pcd).min()
            reward -= 0.2 * min_distance_to_chair  # Penalize distance to chair

        # Stage 2: Push the chair towards the target
        elif distance_to_target > 0.1:  # If the chair is close to the target but not there yet
            # Encourage the robot to push the chair towards the target
            reward -= 0.3 * distance_to_target  # Penalize distance to target

        # Stage 3: Ensure the chair remains stable
        else:  # If the chair is at the target
            # Ensure the chair remains stable
            reward -= 0.2 * chair_tilt  # Penalize chair tilt

        return reward