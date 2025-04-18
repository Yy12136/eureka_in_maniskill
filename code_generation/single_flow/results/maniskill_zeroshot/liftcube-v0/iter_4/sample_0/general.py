import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Check if Cube A is grasped
    if self.agent.check_grasp(self.obj, max_angle=30):
        # Step 2: Check if Cube A is lifted to the goal height
        height_difference = abs(self.obj.pose.p[2] - self.goal_height)
        if height_difference < 0.01:  # Threshold for reaching the goal height
            # Step 3: Check if Cube A is stable
            if check_actor_static(self.obj):
                # Task completed successfully
                reward = 1.0
            else:
                # Penalize instability
                reward = -0.5
        else:
            # Penalize not reaching the goal height
            reward = -0.2
    else:
        # Penalize not grasping Cube A
        reward = -1.0

    return reward