@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    velocity: torch.Tensor,
    actions: torch.Tensor,
    up_vec: torch.Tensor,
    heading_vec: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Device
    device = root_states.device
    
    # Reward temperature parameters
    forward_vel_temp = torch.tensor(0.1, device=device)
    energy_temp = torch.tensor(0.01, device=device)
    alive_bonus_temp = torch.tensor(1.0, device=device)
    
    # Extract forward velocity (project velocity onto heading vector)
    forward_vel = torch.sum(velocity * heading_vec, dim=-1)
    
    # Energy penalty (minimize action magnitude)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Alive bonus (agent gets reward for not falling)
    is_alive = torch.where(torso_position[:, 2] > 0.2, 1.0, 0.0)
    alive_bonus = is_alive.float()
    
    # Forward velocity reward (exponential transformation)
    forward_vel_reward = torch.exp(forward_vel_temp * forward_vel)
    
    # Energy penalty (exponential transformation)
    energy_penalty_reward = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    reward = forward_vel_reward + alive_bonus * alive_bonus_temp + energy_penalty_reward
    
    # Individual reward components
    reward_dict = {
        "forward_vel_reward": forward_vel_reward,
        "alive_bonus": alive_bonus * alive_bonus_temp,
        "energy_penalty_reward": energy_penalty_reward,
    }
    
    return reward, reward_dict
