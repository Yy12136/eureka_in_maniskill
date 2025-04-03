@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor, 
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    angle_temp: float = 5.0  # Increased sensitivity
    position_temp: float = 2.0  # More stable position
    velocity_temp: float = 0.5  # Better scaled
    survival_bonus: float = 0.01  # Small constant bonus
    
    # Extract states
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Angle reward - more sensitive to small deviations
    angle_reward = torch.exp(-angle_temp * torch.abs(pole_angle))
    
    # Position reward - smoother response
    position_reward = 1.0 / (1.0 + position_temp * cart_pos ** 2)
    
    # Velocity penalty - better scaled
    velocity_penalty = 1.0 / (1.0 + velocity_temp * (cart_vel ** 2 + pole_vel ** 2))
    
    # Survival bonus
    survival_reward = torch.ones_like(angle_reward) * survival_bonus
    
    # Combine components with balanced weights
    total_reward = (
        0.5 * angle_reward + 
        0.3 * position_reward + 
        0.1 * velocity_penalty + 
        0.1 * survival_reward
    )
    
    # Store individual components
    reward_dict = {
        "angle_reward": angle_reward,
        "position_reward": position_reward,
        "velocity_penalty": velocity_penalty,
        "survival_reward": survival_reward
    }
    
    return total_reward, reward_dict
