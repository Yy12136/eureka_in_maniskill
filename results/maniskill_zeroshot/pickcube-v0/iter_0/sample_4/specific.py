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
    
    # Step 1: Move end-effector close to the object
    if not grasp_success:
        # Reward for reducing distance between end-effector and object
        dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reward += 1.0 / (1.0 + dist_ee_to_obj)
        
        # Penalize large joint velocities to encourage smooth motion
        joint_velocities = self.agent.robot.get_qvel()[:-2]
        reward -= 0.01 * np.linalg.norm(joint_velocities)
    
    # Step 2: Grasp the object
    if not grasp_success:
        # Reward for successful grasp
        reward += 10.0 if grasp_success else 0.0
    else:
        # Step 3: Move the object towards the goal position
        dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        reward += 1.0 / (1.0 + dist_obj_to_goal)
        
        # Penalize large joint velocities to encourage smooth motion
        joint_velocities = self.agent.robot.get_qvel()[:-2]
        reward -= 0.01 * np.linalg.norm(joint_velocities)
    
    # Step 4: Place the object at the goal position
    if grasp_success and dist_obj_to_goal < 0.01:
        # Reward for successful placement
        reward += 100.0
    
    return reward