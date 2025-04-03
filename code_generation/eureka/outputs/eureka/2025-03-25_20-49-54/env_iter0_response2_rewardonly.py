@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor, 
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward shaping
    angle_temp: float = 0.2
    velocity_temp: float = 0.1
    
    # Cart position (dof_pos[0]) and pole angle (dof_pos[1])
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    
    # Cart velocity (dof_vel[0]) and pole angular velocity (dof_vel[1])
    cart_vel = dof_vel[:, 0]
    pole_vel = dof_vel[:, 1]
    
    # Reward for keeping pole upright (angle close to 0)
    angle_reward = torch.exp(-angle_temp * (pole_angle ** 2))
    
    # Small penalty for cart movement to prevent excessive swinging
    velocity_penalty = torch.exp(-velocity_temp * (cart_vel ** 2))
    
    # Small penalty for pole angular velocity to encourage stability
    angular_vel_penalty = torch.exp(-velocity_temp * (pole_vel ** 2))
    
    # Combine components
    total_reward = angle_reward * velocity_penalty * angular_vel_penalty
    
    # Individual components for debugging
    reward_dict = {
        "angle_reward": angle_reward,
        "velocity_penalty": velocity_penalty,
        "angular_vel_penalty": angular_vel_penalty
    }
    
    return total_reward, reward_dict
