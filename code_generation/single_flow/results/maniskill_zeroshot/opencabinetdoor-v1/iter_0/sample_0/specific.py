import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Stage 1: Approach the Cabinet
    # Reward the robot for moving the base closer to the cabinet
    base_to_cabinet_distance = np.linalg.norm(self.agent.base_pose.p[:2] - self.cabinet.pose.p[:2])
    reward += max(0, 1.0 - base_to_cabinet_distance) * 0.3  # Weight: 0.3

    # Stage 2: Grasp the Handle
    # Reward the robot for aligning the end-effector with the handle
    ee_pose_wrt_handle = self.target_link.pose.inv() * self.agent.hand.pose
    ee_position_error = np.linalg.norm(ee_pose_wrt_handle.p)
    ee_orientation_error = np.linalg.norm(ee_pose_wrt_handle.q - np.array([0, 0, 0, 1]))  # Target orientation is aligned
    reward += max(0, 1.0 - ee_position_error) * 0.2  # Weight: 0.2
    reward += max(0, 1.0 - ee_orientation_error) * 0.1  # Weight: 0.1

    # Reward the robot for closing the gripper around the handle
    gripper_openness_penalty = abs(self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1] - 0.0)  # Fully closed gripper is 0.0
    reward += max(0, 1.0 - gripper_openness_penalty) * 0.1  # Weight: 0.1

    # Stage 3: Open the Door
    # Reward the robot for increasing the cabinet door's qpos
    door_qpos = self.cabinet.qpos[0]  # Assuming the first joint is the door hinge
    target_qpos = self.target_qpos
    qpos_progress = max(0, door_qpos - target_qpos)  # Progress towards opening the door
    reward += qpos_progress * 0.3  # Weight: 0.3

    # Regularization: Penalize large actions to encourage smooth movements
    action_penalty = np.linalg.norm(action)
    reward -= action_penalty * 0.1  # Weight: 0.1

    return reward