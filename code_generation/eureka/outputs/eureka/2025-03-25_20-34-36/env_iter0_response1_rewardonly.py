@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    actions: torch.Tensor,
    heading_vec: torch.Tensor,
    up_vec: torch.Tensor,
    dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    forward_velocity_temp: float = 0.1
    energy_penalty_temp: float = 0.01
    upright_penalty_temp: float = 0.5
    
    # Extract relevant states
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Forward velocity reward (project velocity onto heading vector)
    forward_velocity = torch.sum(velocity * heading_vec, dim=-1)
    forward_reward = torch.exp(forward_velocity_temp * forward_velocity)
    
    # Energy penalty (discourage excessive joint movements)
    energy_penalty = torch.sum(torch.square(actions), dim=-1)
    energy_cost = torch.exp(-energy_penalty_temp * energy_penalty)
    
    # Upright penalty (keep torso upright)
    upright_reward = torch.exp(upright_penalty_temp * up_vec[:, 2])
    
    # Combine rewards
    total_reward = forward_reward * energy_cost * upright_reward
    
    # Individual reward components for debugging
    reward_dict = {
        "forward_reward": forward_reward,
        "energy_cost": energy_cost,
        "upright_reward": upright_reward
    }
    
    return total_reward, reward_dict
