@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the cart position and pole angle from the DOF positions
    cart_pos = dof_pos[:, 0]  # Cart position (x)
    pole_angle = dof_pos[:, 1]  # Pole angle (theta)
    
    # Reward for keeping the pole upright (angle close to 0)
    angle_temperature = 1.0  # Temperature for angle reward
    angle_reward = torch.exp(-angle_temperature * torch.abs(pole_angle))
    
    # Reward for keeping the cart near the center (position close to 0)
    pos_temperature = 0.1  # Temperature for position reward
    pos_reward = torch.exp(-pos_temperature * torch.abs(cart_pos))
    
    # Combine the rewards
    total_reward = angle_reward * pos_reward
    
    # Return the total reward and individual components
    reward_dict = {
        "angle_reward": angle_reward,
        "pos_reward": pos_reward,
    }
    return total_reward, reward_dict
