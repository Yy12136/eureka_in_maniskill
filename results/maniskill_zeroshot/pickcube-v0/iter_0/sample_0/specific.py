import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_move_to_goal = 0.4
    weight_control = 0.2
    weight_stability = 0.1
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successfully grasping the cube
    reward_move_to_goal = 0.0  # Reward for moving the cube closer to the goal
    reward_control = 0.0  # Reward for minimizing joint velocities and smooth control
    reward_stability = 0.0  # Reward for maintaining stability during the task
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    obj_pos = self.obj.pose.p
    goal_pos = self.goal_pos
    
    # Calculate distance between TCP and object
    distance_tcp_to_obj = np.linalg.norm(tcp_pos - obj_pos)
    
    # Calculate distance between object and goal
    distance_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
    
    # Reward for grasping the cube
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0 - distance_tcp_to_obj  # Encourage close proximity before grasping
    
    # Reward for moving the cube closer to the goal
    if self.agent.check_grasp(self.obj):
        reward_move_to_goal = 1.0 - distance_obj_to_goal  # Encourage moving the cube towards the goal
    
    # Reward for minimizing joint velocities and smooth control
    joint_velocities = self.agent.robot.get_qvel()[:-2]
    reward_control = -np.linalg.norm(joint_velocities)  # Penalize high joint velocities
    
    # Reward for maintaining stability (e.g., avoiding large movements)
    joint_positions = self.agent.robot.get_qpos()[:-2]
    reward_stability = -np.linalg.norm(joint_positions)  # Penalize large joint position changes
    
    # Combine all rewards
    reward = (
        weight_grasp * reward_grasp +
        weight_move_to_goal * reward_move_to_goal +
        weight_control * reward_control +
        weight_stability * reward_stability
    )
    
    return reward