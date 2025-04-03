@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    angle_temp: float = 0.5  # Increased sensitivity to angle
    position_temp: float = 0.15  # Slightly increased sensitivity
    velocity_temp: float = 0.02  # Reduced penalty strength
    survival_bonus: float = 0.1  # New component
    
    # Extract states
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Angle reward - more sensitive to deviations
    angle_reward = torch.exp(-angle_temp * torch.abs(pole_angle))
    
    # Position reward - slightly more sensitive
    position_reward = torch.exp(-position_temp * (cart_pos ** 2))
    
    # Velocity penalty - less aggressive
    velocity_penalty = torch.exp(-velocity_temp * (cart_vel ** 2 + 0.5 * pole_vel ** 2))
    
    # Survival bonus - encourages longer episodes
    survival_reward = torch.ones_like(cart_pos) * survival_bonus
    
    # Combine components (multiplicative for main requirements, additive for survival)
    total_reward = angle_reward * position_reward * velocity_penalty + survival_reward
    
    # Store individual components
    reward_dict = {
        "angle_reward": angle_reward,
        "position_reward": position_reward,
        "velocity_penalty": velocity_penalty,
        "survival_bonus": survival_reward
    }
    
    return total_reward, reward_dict
