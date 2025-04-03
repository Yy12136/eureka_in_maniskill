```python
def reward_function(achieved_goal, desired_goal):
    # Sparse reward: 1 if the cube is lifted to the desired goal, 0 otherwise
    reward = 0
    if np.linalg.norm(achieved_goal - desired_goal) < 0.05:  # Threshold for task completion
        reward = 1
    return reward
```