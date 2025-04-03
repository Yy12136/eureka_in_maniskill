@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    angle_temp: float = 0.5  # Increased from 0.2 to be stricter
    position_temp: float = 0.2  # Increased from 0.1
    velocity_temp: float = 0.02  # Reduced from 0.05 to be less harsh
    survival_bonus: float = 0.01  # Small bonus per timestep
    
    # Extract states
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Angle reward - more sensitive to deviations
    angle_reward = torch.exp(-angle_temp * (pole_angle ** 2))
    
    # Position reward - more sensitive to deviations
    position_reward = torch.exp(-position_temp * (cart_pos ** 2))
    
    # Velocity penalty - smoother gradient
    velocity_penalty = 1.0 / (1.0 + velocity_temp * (cart_vel ** 2 + pole_vel ** 2))
    
    # Survival bonus - encourages longer episodes
    survival_reward = torch.ones_like(angle_reward) * survival_bonus
    
    # Combine components
    total_reward = angle_reward * position_reward * velocity_penalty + survival_reward
    
    # Store individual components
    reward_dict = {
        "angle_reward": angle_reward,
        "position_reward": position_reward,
        "velocity_penalty": velocity_penalty,
        "survival_bonus": survival_reward
    }
    
    return total_reward, reward_dict
