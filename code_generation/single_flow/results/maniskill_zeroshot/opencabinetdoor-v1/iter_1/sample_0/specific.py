import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    ee_pose = self.agent.hand.pose
    base_position = self.agent.base_pose.p[:2]
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    base_to_cabinet_dist = np.linalg.norm(base_position - self.cabinet.pose.p[:2])

    # Stage 1: Approach the cabinet
    # Penalize the distance between the robot base and the cabinet
    reward += -0.2 * base_to_cabinet_dist

    # Stage 2: Grasp the handle
    # Penalize the distance between the end-effector and the cabinet handle
    reward += -0.3 * ee_to_handle_dist

    # Encourage the gripper to be closed when near the handle
    if ee_to_handle_dist < 0.05:
        reward += -0.2 * abs(gripper_openness)

    # Stage 3: Open the door
    # Reward the progress of opening the door
    door_progress = max(cabinet_qpos - target_qpos, 0)
    reward += 0.5 * door_progress

    # Stage 4: Task completion
    # Large reward for fully opening the door
    if cabinet_qpos >= target_qpos:
        reward += 1.0

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward