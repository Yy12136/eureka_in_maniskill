import numpy as np

def compute_dense_reward(self, action) -> float:
    weight_distance_ee = 0.4080    # Weight for end-effector distance to cube
    weight_grasp = 0.2534    # Weight for successful grasp
    weight_lift = 0.3386    # Weight for lifting the cube to the desired height
    weight_distance_ee = 0.4080    # Note: weight_distance_ee + weight_grasp + weight_lift = 1.0
    
    # Initialize reward components (total number <= 5)
    reward_distance_ee = 0.0    # Reward for end-effector distance to cube
    reward_grasp = 0.0          # Reward for successful grasp
    reward_lift = 0.0           # Reward for lifting the cube to the desired height
    
    # Calculate reward components
    # 1. End-effector distance to cube
    ee_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    distance_ee_cube = np.linalg.norm(ee_pos - cube_pos)
    reward_distance_ee = 1.0 - np.tanh(5.0 * distance_ee_cube)  # Scale distance to [0, 1]
    
    # 2. Successful grasp
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
    
    # 3. Lifting the cube to the desired height
    desired_height = 0.2
    cube_height = cube_pos[2]
    if self.agent.check_grasp(self.obj):
        reward_lift = 1.0 - np.tanh(10.0 * abs(cube_height - desired_height))  # Scale height difference to [0, 1]
    
    # Combine main rewards
    reward = (
        weight_distance_ee * reward_distance_ee +
        weight_grasp * reward_grasp +
        weight_lift * reward_lift
    )
    
    # Optional: Additional reward components
    # 1. Bonus for maintaining cube above goal height
    if cube_height >= desired_height and self.agent.check_grasp(self.obj):
        reward += 0.1  # Small bonus for maintaining the cube at or above the desired height
    
    # 2. Penalty for large actions (regularization)
    action_penalty = -0.01 * np.linalg.norm(action)  # Penalize large actions to encourage smooth control
    reward += action_penalty
    
    return reward