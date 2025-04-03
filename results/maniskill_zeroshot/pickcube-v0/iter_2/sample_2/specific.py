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
        reward += 2.0  # Higher reward for successful grasp
    
    # Milestone 3: Move the object to the goal
    if grasp_success:
        distance_to_goal = np.linalg.norm(obj_pos - self.goal_pos)
        move_reward = max(0, 3.0 - 0.5 * distance_to_goal)  # Higher reward for moving closer to the goal
        reward += move_reward
    
    # Milestone 4: Place the object at the goal
    if grasp_success and distance_to_goal < 0.05:  # Threshold for considering the object placed
        reward += 5.0  # High reward for placing the object at the goal
    
    # Penalize large joint velocities to encourage smooth motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward -= 0.02 * np.linalg.norm(joint_velocities)
    
    # Penalize large actions to encourage efficient movement
    reward -= 0.02 * np.linalg.norm(action)
    
    # Penalize deviations from the optimal path
    if grasp_success:
        optimal_path_distance = np.linalg.norm(ee_pos - self.goal_pos)
        reward -= 0.01 * optimal_path_distance
    
    return reward