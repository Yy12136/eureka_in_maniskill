import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Distance to Cube A
    ee_pose_wrt_cubeA = self.obj.pose.inv() * self.tcp.pose
    distance_to_cubeA = np.linalg.norm(ee_pose_wrt_cubeA.p)
    reward += 1.0 / (1.0 + distance_to_cubeA)  # Encourage closer proximity

    # Step 2: Grasp Success
    if self.agent.check_grasp(self.obj):
        reward += 1.0  # Reward for successful grasp

    # Step 3: Lift Height
    if self.agent.check_grasp(self.obj):
        cubeA_height = self.obj.pose.p[2]  # Z-coordinate of cube A
        lift_reward = np.clip(cubeA_height - 0.02, 0, 0.1)  # Encourage lifting
        reward += lift_reward

    # Step 4: Distance to Goal Position
    if self.agent.check_grasp(self.obj):
        distance_to_goal = np.linalg.norm(self.obj.pose.p - self.goal_pos)
        reward += 1.0 / (1.0 + distance_to_goal)  # Encourage moving closer to the goal

    # Step 5: Release at Goal
    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for releasing at the goal position

    # Penalties
    if self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 0.5  # Penalize moving cube A away from the goal while grasping

    if not self.agent.check_grasp(self.obj) and np.linalg.norm(self.obj.pose.p - self.goal_pos) > 0.1:
        reward -= 1.0  # Penalize dropping cube A away from the goal

    return reward