import numpy as np
import scipy.spatial.distance

def compute_dense_reward(self, action) -> float:
    # Initialize reward components
    reward = 0.0

    # 1. Distance between chair and target location
    chair_position = self.root_link.pose.p[:2]  # XY position of the chair
    target_position = self.target_xy
    distance_to_target = np.linalg.norm(chair_position - target_position)
    reward += -0.5 * distance_to_target  # Encourage reducing distance to target

    # 2. Chair stability (prevent falling over)
    z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])  # Angle between chair's z-axis and world z-axis
    reward += -0.3 * chair_tilt  # Penalize chair tilting

    # 3. Regularization of robot's action
    action_penalty = 0.1 * np.linalg.norm(action)
    reward += -action_penalty  # Penalize large actions

    # 4. Staged rewards
    if distance_to_target < 0.1:  # Chair is close to the target
        reward += 1.0  # Bonus for reaching the target
        if chair_tilt < 0.1:  # Chair is stable
            reward += 1.0  # Bonus for keeping the chair stable

    # 5. Additional constraints (optional)
    # Ensure the robot's base is not too far from the chair
    base_position = self.robot.base_position
    distance_to_chair = np.linalg.norm(base_position - chair_position)
    reward += -0.2 * distance_to_chair  # Penalize being too far from the chair

    return reward