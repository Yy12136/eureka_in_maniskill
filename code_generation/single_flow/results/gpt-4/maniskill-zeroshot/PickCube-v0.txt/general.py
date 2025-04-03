import numpy as np
from scipy.spatial.distance import cdist

class BaseEnv(gym.Env):
    def compute_dense_reward(self, action) -> float:
        # Initialize the reward
        reward = 0.0
        
        # Get the current pose of the end-effector and cube A
        ee_pose = self.robot.ee_pose.p
        cubeA_pose = self.cubeA.pose.p
        
        # 1. Distance between the robot's gripper and cube A
        dist_ee_to_cubeA = np.linalg.norm(ee_pose - cubeA_pose)
        reward -= 0.5 * dist_ee_to_cubeA  # Encourage the robot to approach cube A
        
        # 2. Grasping reward
        if self.robot.check_grasp(self.cubeA):
            reward += 1.0  # Reward for successfully grasping cube A
            
            # 3. Lifting reward
            lift_height = cubeA_pose[2] - self.cube_half_size  # Height above the table
            if lift_height > 0.05:  # Encourage lifting cube A above a certain height
                reward += 0.5 * lift_height
                
            # 4. Distance between cube A and the goal position
            dist_cubeA_to_goal = np.linalg.norm(cubeA_pose - self.goal_position)
            reward -= 0.5 * dist_cubeA_to_goal  # Encourage moving cube A to the goal position
            
            # 5. Release reward (optional, if the task requires releasing the cube)
            if dist_cubeA_to_goal < 0.02 and self.robot.gripper_openness > 0.8:  # Close to the goal and gripper is open
                reward += 1.0  # Reward for releasing cube A at the goal position
        
        # 6. Action regularization
        action_penalty = 0.01 * np.linalg.norm(action)
        reward -= action_penalty  # Penalize large actions to encourage smooth movements
        
        return reward