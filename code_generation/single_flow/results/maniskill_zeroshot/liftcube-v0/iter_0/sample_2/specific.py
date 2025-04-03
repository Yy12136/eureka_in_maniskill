import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get the current pose of the end-effector and cube A
    ee_pose = self.tcp.pose.p
    cubeA_pose = self.obj.pose.p
    
    # Milestone 1: Approach Cube A
    # Calculate the distance between the end-effector and cube A
    distance_to_cubeA = np.linalg.norm(ee_pose - cubeA_pose)
    # Reward for reducing the distance to cube A
    reward += -distance_to_cubeA * 0.5  # Weight: 0.5
    
    # Milestone 2: Grasp Cube A
    if distance_to_cubeA < 0.05:  # If the gripper is close enough to cube A
        # Check if the gripper is grasping cube A
        is_grasping = self.agent.check_grasp(self.obj, max_angle=30)
        if is_grasping:
            # Reward for successfully grasping cube A
            reward += 1.0  # Weight: 1.0
        else:
            # Penalize for not grasping cube A when close
            reward -= 0.5  # Weight: 0.5
    
    # Milestone 3: Lift Cube A
    if is_grasping:
        # Calculate the height difference between the current and goal height
        current_height = cubeA_pose[2]
        height_difference = self.goal_height - current_height
        # Reward for lifting cube A closer to the goal height
        reward += -height_difference * 0.5  # Weight: 0.5
        
        # Milestone 4: Stabilize Cube A
        # Check if cube A is static (i.e., not moving)
        is_static = check_actor_static(self.obj)
        if is_static:
            # Reward for keeping cube A stable
            reward += 0.5  # Weight: 0.5
    
    # Regularization of the robot's action
    # Penalize large actions to encourage smoother movements
    action_penalty = np.linalg.norm(action) * 0.1
    reward -= action_penalty  # Weight: 0.1
    
    return reward