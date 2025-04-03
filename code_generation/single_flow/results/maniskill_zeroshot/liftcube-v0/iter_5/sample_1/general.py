import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if self.agent.check_grasp(self.obj, max_angle=30):  # Check if Cube A is grasped
        grasp_reward = 1.0  # Reward for successful grasp
        reward += grasp_reward

        # Step 2: Lift Cube A to the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)
        if height_difference < 0.01:  # Threshold for reaching the goal height
            lift_reward = 1.0  # Reward for lifting Cube A to the goal height
            reward += lift_reward

            # Step 3: Stabilize Cube A
            if check_actor_static(self.obj):  # Check if Cube A is stable
                stabilize_reward = 1.0  # Reward for stabilization
                reward += stabilize_reward

    return reward