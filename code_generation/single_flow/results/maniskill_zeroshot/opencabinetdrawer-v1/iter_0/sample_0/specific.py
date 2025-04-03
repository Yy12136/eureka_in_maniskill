import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Stage 1: Navigation to the cabinet
    # Reward for reducing the distance between the robot base and the cabinet
    base_to_cabinet_distance = np.linalg.norm(self.agent.base_pose.p[:2] - self.cabinet.pose.p[:2])
    navigation_reward = -base_to_cabinet_distance  # Encourage closer proximity

    # Stage 2: Grasping the handle
    # Reward for reducing the distance between the gripper and the handle
    ee_coords = self.agent.get_ee_coords()  # Get 3D positions of the gripper fingers
    handle_pcd = transform_points(self.target_link.pose.to_transformation_matrix(), self.target_handle_pcd)  # Get point cloud of the handle
    gripper_to_handle_distance = cdist(ee_coords, handle_pcd).min(-1).mean()  # Mean distance to handle
    grasping_reward = -gripper_to_handle_distance  # Encourage closer proximity

    # Reward for aligning the gripper with the handle
    ee_pose_wrt_handle = self.target_link.pose.inv() * self.agent.hand.pose
    alignment_error = np.linalg.norm(ee_pose_wrt_handle.p)  # Position error
    alignment_reward = -alignment_error  # Encourage better alignment

    # Reward for closing the gripper
    gripper_openness = self.agent.robot.get_qpos()[-1] / self.agent.robot.get_qlimits()[-1, 1]
    gripper_closure_reward = -gripper_openness  # Encourage gripper to close

    # Stage 3: Pulling the drawer
    # Reward for increasing the drawer's qpos
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    drawer_progress = max(0, drawer_qpos - target_qpos)  # Progress toward target
    pulling_reward = drawer_progress  # Encourage pulling the drawer

    # Stage 4: Task completion bonus
    # Bonus reward if the drawer is fully opened
    task_complete_bonus = 10.0 if drawer_qpos >= target_qpos else 0.0

    # Regularization of the robot's action
    action_penalty = -0.01 * np.linalg.norm(action)  # Penalize large actions

    # Combine rewards with appropriate weights
    reward = (
        0.3 * navigation_reward +
        0.4 * grasping_reward +
        0.2 * alignment_reward +
        0.1 * gripper_closure_reward +
        0.5 * pulling_reward +
        task_complete_bonus +
        action_penalty
    )

    return reward