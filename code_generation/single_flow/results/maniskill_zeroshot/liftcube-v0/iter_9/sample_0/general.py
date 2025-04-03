import numpy as np

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Grasp Cube A
    if not self.agent.check_grasp(self.obj, max_angle=30):
        # No reward until the cube is grasped
        return reward
    
    # Step 2: Lift Cube A to the goal height
    height_difference = abs(self.obj.pose.p[2] - self.goal_height)
    if height_difference < 0.01:  # Threshold for reaching the goal height
        # Large reward for successfully lifting the cube
        reward += 1.0

    return reward