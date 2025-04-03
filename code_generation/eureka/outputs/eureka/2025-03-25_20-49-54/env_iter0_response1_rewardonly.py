@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor, 
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    angle_temp: float = 5.0
    position_temp: float = 1.0
    velocity_temp: float = 0.1
    
    # Extract cart position (dof_pos[0]) and pole angle (dof_pos[1])
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    
    # Extract velocities
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Angle reward - maximize when pole is upright (angle = 0)
    angle_reward = torch.exp(-angle_temp * torch.square(pole_angle))
    
    # Position reward - keep cart near center
    position_reward = torch.exp(-position_temp * torch.square(cart_pos))
    
    # Velocity penalty - minimize velocities
    velocity_penalty = torch.square(cart_vel) + torch.square(pole_vel)
    velocity_reward = torch.exp(-velocity_temp * velocity_penalty)
    
    # Combine rewards with weights
    total_reward = 0.5 * angle_reward + 0.3 * position_reward + 0.2 * velocity_reward
    
    # Individual components for logging
    reward_dict = {
        "angle_reward": angle_reward,
        "position_reward": position_reward,
        "velocity_reward": velocity_reward
    }
    
    return total_reward, reward_dict
