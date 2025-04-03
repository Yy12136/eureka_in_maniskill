import numpy as np
from scipy.spatial.distance import cdist

def compute_dense_reward(self, action) -> float:
    # Initialize reward
    reward = 0.0

    # Step 1: Move gripper to Cube A
    gripper_to_cubeA_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeA.pose.p)
    reward += -0.5 * gripper_to_cubeA_dist  # Penalize distance to Cube A

    # Step 2: Grasp Cube A
    if gripper_to_cubeA_dist < 0.05:  # If gripper is close to Cube A
        if self.robot.check_grasp(self.cubeA, max_angle=30):  # Check if Cube A is grasped
            reward += 1.0  # Reward for successful grasp
        else:
            reward += -0.2 * (1 - self.robot.gripper_openness)  # Penalize gripper openness if not grasping

    # Step 3: Lift Cube A and move it to Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is grasped
        cubeA_to_cubeB_dist = np.linalg.norm(self.cubeA.pose.p - self.cubeB.pose.p)
        reward += -0.5 * cubeA_to_cubeB_dist  # Penalize distance to Cube B

        # Ensure Cube A is lifted above Cube B
        if self.cubeA.pose.p[2] < self.cubeB.pose.p[2] + 2 * self.cube_half_size:
            reward += -0.3 * (self.cubeB.pose.p[2] + 2 * self.cube_half_size - self.cubeA.pose.p[2])

    # Step 4: Place Cube A on Cube B
    if self.robot.check_grasp(self.cubeA, max_angle=30):  # If Cube A is grasped
        if cubeA_to_cubeB_dist < 0.05:  # If Cube A is close to Cube B
            reward += 1.0  # Reward for placing Cube A near Cube B

            # Ensure Cube A is stable on Cube B
            if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
                reward += 2.0  # Reward for stable placement and release

    # Step 5: Penalize action magnitude for smooth movements
    reward += -0.01 * np.linalg.norm(action)

    # Step 6: Additional reward for aligning Cube A with Cube B
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_center = self.cubeA.pose.p
        cubeB_center = self.cubeB.pose.p
        horizontal_dist = np.linalg.norm(cubeA_center[:2] - cubeB_center[:2])
        if horizontal_dist < 0.01:  # If Cube A is centered on Cube B
            reward += 0.5  # Reward for precise alignment

    # Step 7: Penalize excessive joint velocities for smoother movements
    reward += -0.01 * np.linalg.norm(self.robot.qvel)

    # Step 8: Penalize excessive angular velocity of Cube A for stability
    if self.cubeA.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeA.angular_velocity)

    # Step 9: Reward for keeping Cube A upright
    if self.cubeA.check_static():
        cubeA_rotation = self.cubeA.pose.to_transformation_matrix()[:3, :3]
        desired_rotation = np.eye(3)  # Desired rotation is upright
        rotation_diff = np.linalg.norm(cubeA_rotation - desired_rotation)
        reward += -0.1 * rotation_diff  # Penalize deviation from upright orientation

    # Step 10: Reward for minimizing the time taken to complete the task
    if self.cubeA.check_static() and not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 0.1 * (1 - self.current_step / self.max_steps)  # Reward for completing the task quickly

    # Step 11: Penalize any unnecessary movement of Cube B
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.velocity)  # Penalize Cube B's velocity

    # Step 12: Reward for maintaining a stable grasp during movement
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += 0.05 * (1 - np.linalg.norm(self.cubeA.velocity))  # Reward for stable grasp during movement

    # Step 13: Reward for minimizing the tilt of Cube A during placement
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        cubeA_rotation = self.cubeA.pose.to_transformation_matrix()[:3, :3]
        desired_rotation = np.eye(3)  # Desired rotation is upright
        rotation_diff = np.linalg.norm(cubeA_rotation - desired_rotation)
        reward += -0.05 * rotation_diff  # Penalize tilt during placement

    # Step 14: Reward for minimizing the distance between the gripper and Cube A during grasping
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += -0.1 * gripper_to_cubeA_dist  # Penalize distance during grasping

    # Step 15: Reward for minimizing the distance between the gripper and Cube B during placement
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        gripper_to_cubeB_dist = np.linalg.norm(self.robot.ee_pose.p - self.cubeB.pose.p)
        reward += -0.1 * gripper_to_cubeB_dist  # Penalize distance during placement

    # Step 16: Reward for minimizing the distance between Cube A and Cube B during placement
    if self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += -0.1 * cubeA_to_cubeB_dist  # Penalize distance during placement

    # Step 17: Reward for minimizing the distance between the gripper and Cube A after placement
    if not self.robot.check_grasp(self.cubeA, max_angle=30):
        reward += -0.1 * gripper_to_cubeA_dist  # Penalize distance after placement

    # Step 18: Reward for minimizing the angular velocity of Cube B for stability
    if self.cubeB.check_static():
        reward += -0.01 * np.linalg.norm(self.cubeB.angular_velocity)  # Penalize Cube B's angular velocity

    # Step 19: Reward for minimizing the tilt of Cube B for stability
    if self.cubeB.check_static():
        cubeB_rotation = self.cubeB.pose.to_transformation_matrix()[:3, :3]
        desired_rotation = np.eye(3)  # Desired rotation is upright
        rotation_diff = np.linalg.norm(cubeB_rotation - desired_rotation)
        reward += -0.05 * rotation_diff  # Penalize tilt of Cube B

    return reward