import numpy as np

def compute_dense_reward(self, action) -> float:
                # Initialize reward
                reward = 0.0
                
                # Get positions
                ee_pos = self.tcp.pose.p
                obj_pos = self.obj.pose.p
                
                # Check grasp
                grasp_success = self.agent.check_grasp(self.obj)
                
                # Step 1: Move end-effector close to the object
                if not grasp_success:
                    distance_to_obj = np.linalg.norm(ee_pos - obj_pos)
                    reward += 1.0 / (1.0 + distance_to_obj)  # Encourage proximity
                
                # Step 2: Grasp the object
                if grasp_success:
                    reward += 10.0  # Large reward for successful grasp
                
                # Step 3: Move the object to the goal position
                if grasp_success:
                    distance_to_goal = np.linalg.norm(obj_pos - self.goal_pos)
                    reward += 1.0 / (1.0 + distance_to_goal)  # Encourage moving towards goal
                
                # Step 4: Place the object at the goal position
                if grasp_success and np.linalg.norm(obj_pos - self.goal_pos) < 0.01:
                    reward += 20.0  # Large reward for placing the object at the goal
                
                # Penalize large joint velocities for smooth motion
                joint_velocities = self.agent.robot.get_qvel()[:-2]
                reward -= 0.01 * np.linalg.norm(joint_velocities)  # Penalize high velocities
                
                return reward