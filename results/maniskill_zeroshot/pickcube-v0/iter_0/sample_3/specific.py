import numpy as np

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get positions
    ee_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    
    # Check grasp
    grasp_success = self.agent.check_grasp(self.obj)
    
    # Milestone 1: Reach the cube
    distance_to_cube = np.linalg.norm(ee_pos - obj_pos)
    reward += -0.1 * distance_to_cube  # Encourage the end-effector to approach the cube
    
    # Milestone 2: Grasp the cube
    if grasp_success:
        reward += 1.0  # Reward for successfully grasping the cube
    else:
        reward += -0.5 * (1 - grasp_success)  # Penalize if the cube is not grasped
    
    # Milestone 3: Move the cube to the goal position
    if grasp_success:
        distance_to_goal = np.linalg.norm(obj_pos - self.goal_pos)
        reward += -0.1 * distance_to_goal  # Encourage moving the cube closer to the goal
    
    # Milestone 4: Place the cube at the goal position
    if grasp_success and np.linalg.norm(obj_pos - self.goal_pos) < 0.01:
        reward += 2.0  # Reward for placing the cube at the goal position
    
    # Penalize large joint velocities to encourage smooth motion
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward += -0.01 * np.linalg.norm(joint_velocities)
    
    return reward