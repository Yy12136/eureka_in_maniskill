@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor, 
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    angle_temp: float = 0.2
    position_temp: float = 0.1
    velocity_temp: float = 0.05
    
    # Extract cart position (dof_pos[0]) and pole angle (dof_pos[1])
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    
    # Extract velocities
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Reward for keeping pole upright (angle close to 0)
    angle_reward = torch.exp(-angle_temp * (pole_angle ** 2))
    
    # Reward for keeping cart near center
    position_reward = torch.exp(-position_temp * (cart_pos ** 2))
    
    # Penalty for excessive velocities (both cart and pole)
    velocity_penalty = torch.exp(-velocity_temp * (cart_vel ** 2 + pole_vel ** 2))
    
    # Combine components
    total_reward = angle_reward * position_reward * velocity_penalty
    
    # Store individual components for debugging/analysis
    reward_dict = {
        "angle_reward": angle_reward,
        "position_reward": position_reward,
        "velocity_penalty": velocity_penalty
    }
    
    return total_reward, reward_dict
