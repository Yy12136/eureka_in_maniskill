import numpy as np

def compute_dense_reward(self, action):
    reward = 0.0

    # Task completion condition
    is_drawer_open = self.cabinet_drawer.qpos >= self.target_qpos

    # Sparse reward for task completion
    if is_drawer_open:
        reward += 10.0  # Large reward for successfully opening the drawer
        return reward

    # Reaching reward: Encourage the robot to move towards the drawer handle
    gripper_pos = self.robot.ee_pose.p
    handle_pos = self.cabinet_drawer.handle_pose.p
    dist_to_handle = np.linalg.norm(gripper_pos - handle_pos)
    reaching_reward = 1 - np.tanh(5 * dist_to_handle)
    reward += reaching_reward

    # Grasping reward: Encourage the robot to grasp the handle
    is_grasped = self.robot.check_grasp(self.cabinet_drawer.handle)
    if is_grasped:
        reward += 1.0

    # Pulling reward: Encourage the robot to pull the drawer open
    if is_grasped:
        current_qpos = self.cabinet_drawer.qpos
        pulling_reward = min(current_qpos / self.target_qpos, 1.0)
        reward += pulling_reward

    return reward