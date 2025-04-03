@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, heading_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters for reward components
    forward_vel_temp: float = 0.1
    energy_penalty_temp: float = 0.01
    
    # Extract velocity (linear and angular)
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    
    # Forward velocity is the component of velocity in the heading direction
    forward_vel = torch.sum(velocity * heading_vec, dim=-1)
    
    # Forward velocity reward (exponentially scaled)
    forward_reward = torch.exp(forward_vel_temp * forward_vel)
    
    # Energy penalty (sum of squared actions)
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_cost = torch.exp(-energy_penalty_temp * energy_penalty)
    
    # Total reward
    total_reward = forward_reward * energy_cost
    
    # Individual reward components
    reward_dict = {
        "forward_reward": forward_reward,
        "energy_cost": energy_cost,
    }
    
    return total_reward, reward_dict
