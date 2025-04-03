import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Get relevant variables
    cabinet_handle = self.target_link
    ee_pose = self.agent.hand.pose
    base_position = self.agent.base_pose.p[:2]
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    cabinet_qpos = cabinet_handle.qpos
    target_qpos = cabinet_handle.target_qpos

    # Milestone 1: Distance between robot base and cabinet
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)

    # Milestone 2: Distance between end-effector and cabinet handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()

    # Milestone 3: Door opening progress
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos

    # Milestone 4: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
        return reward  # Early return to focus on task completion

    # Sparse rewards for key milestones
    if base_to_cabinet_dist < 0.1:
        reward += 1.0  # Reward for approaching the cabinet

    if ee_to_handle_dist < 0.05:
        reward += 2.0  # Reward for grasping the handle

    if door_progress > 0:
        reward += 3.0  # Reward for starting to open the door

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    return reward