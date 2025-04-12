import numpy as np

def compute_dense_reward(self, action) -> float:
    # Define reward weights
    weight_grasp = 0.3
    weight_lift = 0.4
    weight_stability = 0.3
    
    # Initialize components
    reward_grasp = 0.0  # Reward for successful grasp
    reward_lift = 0.0   # Reward for lifting the cube
    reward_stability = 0.0  # Reward for maintaining stability during the lift
    
    # Get positions
    tcp_pos = self.tcp.pose.p
    cube_pos = self.obj.pose.p
    
    # Calculate distance between TCP and cube
    distance = np.linalg.norm(tcp_pos - cube_pos)
    
    # Reward for approaching the cube
    if distance < 0.02 * 2:
        reward_grasp = 1.0 - (distance / (0.02 * 2))
    
    # Check if the cube is grasped
    if self.agent.check_grasp(self.obj):
        reward_grasp = 1.0
        
        # Calculate the height difference
        lift_height = cube_pos[2] - 0.02
        target_height = 0.2
        
        # Reward for lifting the cube
        if lift_height > 0:
            reward_lift = min(lift_height / target_height, 1.0)
        
        # Reward for maintaining stability during the lift
        if check_actor_static(self.obj):
            reward_stability = 1.0
    
    # Combine all rewards
    reward = weight_grasp * reward_grasp + weight_lift * reward_lift + weight_stability * reward_stability
    
    return reward