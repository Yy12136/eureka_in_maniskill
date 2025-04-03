import numpy as np

def reward_function(cube_height, is_grasped, is_held, time_step):
    # Parameters
    target_height = 0.2
    halfway_height = 0.1
    hold_duration = 0.5  # seconds

    # Initialize reward
    reward = 0

    # Check for successful grasp
    if is_grasped and not previously_grasped:
        reward += 0.5

    # Check for halfway lift
    if cube_height >= halfway_height and not previously_halfway:
        reward += 0.5

    # Check for final success
    if cube_height >= target_height and is_held_for(hold_duration):
        reward += 1

    # Penalties
    if is_grasped and cube_height < 0.01:  # Cube dropped
        reward -= 0.1
    if time_step > max_steps and not is_grasped:  # Failed to grasp
        reward -= 0.1

    return reward