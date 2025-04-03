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
    reward += max(0, 1.5 - 0.5 * distance_to_obj)  # Stronger encouragement to move closer to the object
    
    # Milestone 2: Grasp the object
    if grasp_success:
        reward += 2.0  # Higher reward for successful grasp
    
    # Milestone 3: Move the object to the goal
    if grasp_success:
        distance_to_goal = np.linalg.norm(obj_pos - self.goal_pos)
        reward += max(0, 3.0 - 0.5 * distance_to_goal)  # Stronger encouragement to move the object closer to the goal
    
    # Milestone 4: Object placed at goal
    if grasp_success and distance_to_goal < 0.05:  # Threshold for successful placement
        reward += 5.0  # High reward for completing the task
    
    # Penalize large joint velocities to encourage smooth motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward -= 0.02 * np.linalg.norm(joint_velocities)
    
    # Penalize large actions to encourage efficient movement
    reward -= 0.02 * np.linalg.norm(action)
    
    # Penalize excessive distance between end-effector and object if not grasping
    if not grasp_success:
        reward -= 0.01 * distance_to_obj
    
    return reward