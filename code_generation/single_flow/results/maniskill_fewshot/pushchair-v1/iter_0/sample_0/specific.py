import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    reward = 0.0

    # Chair and target positions
    chair_pos = self.chair.pose.p[:2]  # XY position of the chair
    target_pos = self.target_xy  # XY target position
    chair_to_target_dist = np.linalg.norm(chair_pos - target_pos)

    # Chair stability
    z_axis_chair = self.chair.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])  # Angle between chair's z-axis and world z-axis
    max_tilt = np.pi / 6  # Maximum allowed tilt (30 degrees)
    is_chair_stable = chair_tilt <= max_tilt

    # Robot's end-effector positions
    ee_coords = self.robot.get_ee_coords()  # 4x3 array of end-effector finger positions
    chair_pcd = self.chair.get_pcd()  # Point cloud of the chair

    # Distance between robot's end-effectors and the chair
    ee_to_chair_dist = cdist(ee_coords, chair_pcd).min(axis=1).mean()

    # Staged rewards
    if chair_to_target_dist <= 0.05:  # Chair is close to the target
        if self.chair.check_static() and is_chair_stable:  # Chair is stable and static
            reward += 10.0  # Success reward
            return reward
        else:
            reward += 5.0  # Chair is at the target but not stable
    else:
        # Reward for pushing the chair toward the target
        prev_chair_to_target_dist = getattr(self, "prev_chair_to_target_dist", chair_to_target_dist)
        chair_movement_reward = prev_chair_to_target_dist - chair_to_target_dist
        reward += 2.0 * chair_movement_reward
        self.prev_chair_to_target_dist = chair_to_target_dist  # Update for the next step

        # Reward for getting closer to the chair
        reaching_reward = 1 - np.tanh(5.0 * ee_to_chair_dist)
        reward += reaching_reward

    # Penalize chair instability
    if not is_chair_stable:
        reward -= 5.0 * (chair_tilt / max_tilt)  # Penalize based on tilt severity

    # Regularization of robot's action
    action_penalty = 0.01 * np.linalg.norm(action)
    reward -= action_penalty

    return reward