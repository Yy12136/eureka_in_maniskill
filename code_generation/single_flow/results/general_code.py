```python
def reward_function(self):
    reward = 0.0
    
    # Step 1: Move end-effector close to the cube
    ee_to_cube_distance = np.linalg.norm(self.robot.ee_pose[:3] - self.cubeA.position)
    reward += max(0, 1.0 - ee_to_cube_distance) * 0.3
    
    # Step 2: Orient end-effector for grasping
    ee_orientation_diff = np.linalg.norm(self.robot.ee_pose[3:] - [0, 0, 0, 1])  # Assuming optimal orientation is identity quaternion
    reward += max(0, 1.0 - ee_orientation_diff) * 0.2
    
    # Step 3: Grasp the cube
    if self.robot.check_grasp(self.cubeA):
        reward += 0.5  # Milestone reward for successful grasp
    
    # Step 4: Lift the cube
    if self.robot.check_grasp(self.cubeA):
        cube_height = self.cubeA.position[2]
        reward += cube_height * 0.2  # Reward proportional to height
    
    # Step 5: Ensure the cube is static after lifting
    if self.cubeA.check_static() and self.robot.check_grasp(self.cubeA):
        reward += 0.3  # Milestone reward for stable lifting
    
    # Penalize excessive joint velocities for smooth motion
    joint_velocity_penalty = -0.01 * np.linalg.norm(self.robot.qvel)
    reward += joint_velocity_penalty
    
    return reward
```