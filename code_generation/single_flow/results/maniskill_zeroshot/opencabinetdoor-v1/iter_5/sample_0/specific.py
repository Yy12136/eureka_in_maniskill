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

    # Milestone 1: Distance between robot base and cabinet
    cabinet_position = self.cabinet.pose.p[:2]  # Only consider XY plane for mobile base
    base_to_cabinet_dist = np.linalg.norm(base_position - cabinet_position)
    reward += -0.1 * base_to_cabinet_dist  # Penalize distance to cabinet

    # Milestone 2: Distance between end-effector and cabinet handle
    ee_coords = self.agent.get_ee_coords()
    handle_pcd = cabinet_handle.get_world_pcd()
    ee_to_handle_dist = cdist(ee_coords, handle_pcd).min()
    reward += -0.2 * ee_to_handle_dist  # Penalize distance to handle

    # Milestone 3: Gripper openness
    # Gripper should be closed (value close to 0) when near the handle
    if ee_to_handle_dist < 0.05:  # If gripper is close to handle
        reward += -0.1 * abs(gripper_openness)  # Penalize incorrect gripper openness

    # Milestone 4: Door opening progress
    door_progress = max(cabinet_qpos - target_qpos, 0)  # Positive if qpos > target_qpos
    reward += 0.3 * door_progress  # Reward door opening progress

    # Milestone 5: Task completion
    if cabinet_qpos >= target_qpos:
        reward += 1.0  # Large reward for task completion

    # Regularization: Penalize large actions
    action_penalty = -0.01 * np.linalg.norm(action)
    reward += action_penalty

    # Staged reward: Encourage the robot to complete each milestone in sequence
    if base_to_cabinet_dist < 0.1:
        reward += 0.1  # Bonus for approaching the cabinet
        if ee_to_handle_dist < 0.05:
            reward += 0.2  # Bonus for grasping the handle
            if door_progress > 0:
                reward += 0.3  # Bonus for starting to open the door
                if cabinet_qpos >= target_qpos:
                    reward += 0.4  # Bonus for completing the task

    # Penalty for excessive base movement to encourage efficient navigation
    base_velocity_penalty = -0.01 * np.linalg.norm(self.agent.base_velocity)
    reward += base_velocity_penalty

    # Penalty for excessive joint movement to encourage smooth motion
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.agent.robot.qvel)
    reward += joint_velocity_penalty

    # Orientation consideration: Penalize incorrect end-effector orientation relative to the handle
    handle_pose = cabinet_handle.pose
    ee_orientation_diff = np.linalg.norm(ee_pose.q - handle_pose.q)
    if ee_to_handle_dist < 0.05:
        reward += -0.1 * ee_orientation_diff  # Penalize incorrect orientation when close to the handle

    return reward