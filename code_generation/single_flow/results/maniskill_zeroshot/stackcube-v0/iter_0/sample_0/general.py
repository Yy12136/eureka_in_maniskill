import numpy as np

def reward_function(state):
    # State variables
    cube_A_on_B = state['cube_A_on_B']  # Boolean: True if cube A is stably on cube B
    gripper_grasping = state['gripper_grasping']  # Boolean: True if gripper is grasping cube A
    task_complete = state['task_complete']  # Boolean: True if task is completed
    task_failed = state['task_failed']  # Boolean: True if task fails

    # Reward logic
    if task_complete and cube_A_on_B and not gripper_grasping:
        return 2  # Successful completion
    elif task_failed:
        return -1  # Task failed
    else:
        return 0  # Intermediate state