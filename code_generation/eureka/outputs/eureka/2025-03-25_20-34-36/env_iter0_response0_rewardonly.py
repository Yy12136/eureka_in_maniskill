@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    dof_vel: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Temperature parameters for reward components
    forward_vel_temp: float = 0.1
    energy_temp: float = 0.01
    upright_temp: float = 0.5
    
    # Extract components from root states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    
    # Forward velocity reward (aligned with heading direction)
    forward_vel = torch.sum(velocity * heading_vec, dim=-1)
    forward_vel_reward = torch.exp(forward_vel_temp * forward_vel)
    
    # Energy efficiency penalty (discourage excessive joint movements)
    energy_penalty = torch.sum(torch.square(actions), dim=-1) + 0.05 * torch.sum(torch.square(dof_vel), dim=-1)
    energy_reward = torch.exp(-energy_temp * energy_penalty)
    
    # Upright posture reward (keep torso upright)
    upright_reward = torch.exp(upright_temp * up_vec[:, 2])
    
    # Combine rewards
    total_reward = forward_vel_reward * energy_reward * upright_reward
    
    # Individual reward components for debugging
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "energy_reward": energy_reward,
        "upright_reward": upright_reward
    }
    
    return total_reward, reward_dict
