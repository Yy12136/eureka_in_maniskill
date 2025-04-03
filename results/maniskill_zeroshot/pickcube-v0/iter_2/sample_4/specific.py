import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Reach the object
    distance_to_obj = np.linalg.norm(ee_pos - obj_pos)
    reach_reward = max(0, 1.0 - 0.5 * distance_to_obj)  # Encourage moving closer to the object
    reward += reach_reward
    
    # Milestone 2: Grasp the object
    if grasp_success:
        grasp_reward = 2.0  # Higher reward for successful grasp
        reward += grasp_reward
    else:
        # Penalize for not grasping the object
        reward -= 0.5 * (1.0 - reach_reward)  # Penalize based on how close the end-effector is to the object
    
    # Milestone 3: Move the object to the goal
    if grasp_success:
        distance_to_goal = np.linalg.norm(obj_pos - self.goal_pos)
        move_reward = max(0, 3.0 - 0.5 * distance_to_goal)  # Encourage moving the object closer to the goal
        reward += move_reward
    else:
        # Penalize for not moving the object towards the goal
        reward -= 0.5 * (1.0 - reach_reward)
    
    # Milestone 4: Align the end-effector with the object before grasping
    if not grasp_success:
        alignment_error = np.linalg.norm(ee_pos - obj_pos)
        alignment_reward = max(0, 1.0 - 0.5 * alignment_error)
        reward += alignment_reward
    
    # Penalize large joint velocities to encourage smooth motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward -= 0.01 * np.linalg.norm(joint_velocities)
    
    # Penalize large actions to encourage efficient movement
    reward -= 0.01 * np.linalg.norm(action)
    
    return reward