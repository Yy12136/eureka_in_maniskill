import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    goal_pos = self.goal_pos
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Move end-effector close to the object
    distance_to_obj = np.linalg.norm(ee_pos - obj_pos)
    reward += max(0, 1.0 - distance_to_obj)  # Reward closer proximity
    
    # Milestone 2: Grasp the object
    if grasp_success:
        reward += 10.0  # Large reward for successful grasp
    else:
        reward -= 0.1 * distance_to_obj  # Penalize staying far without grasping
    
    # Milestone 3: Move the object towards the goal
    if grasp_success:
        distance_to_goal = np.linalg.norm(obj_pos - goal_pos)
        reward += max(0, 1.0 - distance_to_goal)  # Reward closer proximity to goal
        
        # Bonus for moving the object directly towards the goal
        direction_to_goal = (goal_pos - obj_pos) / (distance_to_goal + 1e-6)
        obj_velocity = self.obj.pose.p - self.obj.prev_pos
        reward += 0.1 * np.dot(obj_velocity, direction_to_goal)
    
    # Penalize large joint velocities for smoother motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward -= 0.01 * np.linalg.norm(joint_velocities)
    
    # Penalize large actions for energy efficiency
    reward -= 0.001 * np.linalg.norm(action)
    
    return reward