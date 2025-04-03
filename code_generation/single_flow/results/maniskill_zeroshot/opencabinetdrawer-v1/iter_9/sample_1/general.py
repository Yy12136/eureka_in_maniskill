import numpy as np
from scipy.spatial.distance import cdist

def compute_sparse_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0
    
    # Get relevant variables
    drawer_qpos = self.link_qpos
    target_qpos = self.target_qpos
    
    # Stage 1: Task Completion
    # Large reward for completing the task
    if drawer_qpos >= target_qpos:
        reward += 10.0  # Large reward for task completion
    
    # Stage 2: Penalize for pushing the drawer instead of pulling
    if drawer_qpos < 0 and np.abs(drawer_qpos) > 0.01:
        reward -= 1.0  # Penalize for pushing the drawer
    
    # Stage 3: Penalize if the robot base moves too much
    base_movement_penalty = -0.1 * np.linalg.norm(self.agent.base_link.velocity[:2])
    reward += base_movement_penalty
    
    # Stage 4: Penalize if the robot base is too far from the cabinet
    base_to_cabinet_dist = np.linalg.norm(self.agent.base_pose.p[:2] - self.cabinet.pose.p[:2])
    if base_to_cabinet_dist > 0.5:
        reward -= 1.0  # Penalize for being too far from the cabinet
    
    # Stage 5: Penalize if the robot is not facing the cabinet
    robot_to_cabinet_vector = self.cabinet.pose.p[:2] - self.agent.base_pose.p[:2]
    robot_forward_vector = np.array([np.cos(self.agent.hand.pose.q[0]), np.sin(self.agent.hand.pose.q[0])])
    angle_diff = np.arccos(np.clip(np.dot(robot_to_cabinet_vector, robot_forward_vector) / (np.linalg.norm(robot_to_cabinet_vector) * np.linalg.norm(robot_forward_vector)), -1.0, 1.0))
    if angle_diff > np.pi / 4:  # If the robot is not facing the cabinet within 45 degrees
        reward -= 1.0  # Penalize for not facing the cabinet
    
    return reward