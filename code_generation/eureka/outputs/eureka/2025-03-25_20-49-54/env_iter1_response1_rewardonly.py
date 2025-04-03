@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters
    angle_temp: float = 0.1  # More sensitive to small angle deviations
    position_temp: float = 0.5  # Stronger penalty for position deviations
    velocity_temp: float = 0.01  # More aggressive velocity penalty
    
    # Component weights
    angle_weight: float = 0.5
    position_weight: float = 0.3
    velocity_weight: float = 0.2
    
    # Extract states
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Angle reward - more sensitive version
    angle_reward = torch.exp(-angle_temp * (pole_angle ** 2))
    
    # Position penalty - quadratic term for stronger correction
    position_penalty = 1.0 - torch.min(position_temp * (cart_pos ** 2), torch.ones_like(cart_pos))
    
    # Velocity penalty - more aggressive version
    vel_magnitude = cart_vel ** 2 + pole_vel ** 2
    velocity_penalty = torch.exp(-velocity_temp * vel_magnitude)
    
    # Survival bonus (linear with time)
    survival_bonus = 0.001
    
    # Weighted components
    total_reward = (
        angle_weight * angle_reward +
        position_weight * position_penalty +
        velocity_weight * velocity_penalty +
        survival_bonus
    )
    
    reward_dict = {
        "angle_reward": angle_reward,
        "position_penalty": position_penalty,
        "velocity_penalty": velocity_penalty,
        "survival_bonus": torch.ones_like(total_reward) * survival_bonus
    }
    
    return total_reward, reward_dict
