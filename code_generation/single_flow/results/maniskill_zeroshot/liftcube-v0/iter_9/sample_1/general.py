import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if not self.agent.check_grasp(self.obj, max_angle=30):
        # Penalize if Cube A is not grasped
        reward -= 1.0
    else:
        # Reward for successful grasp
        reward += 1.0

        # Step 2: Lift Cube A to the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)
        if height_difference < 0.01:  # Threshold for reaching the goal height
            # Reward for lifting Cube A to the goal height
            reward += 1.0

            # Step 3: Stabilize Cube A
            if check_actor_static(self.obj):  # Check if Cube A is static
                # Reward for stabilization
                reward += 1.0
            else:
                # Penalize instability
                reward -= 0.5
        else:
            # Penalize not reaching the goal height
            reward -= 0.5

    return reward