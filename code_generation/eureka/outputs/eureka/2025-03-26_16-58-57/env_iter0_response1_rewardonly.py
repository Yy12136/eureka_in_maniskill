@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract cart position (dof_pos[:, 0]) and pole angle (dof_pos[:, 1])
    cart_pos = dof_pos[:, 0]
    pole_angle = dof_pos[:, 1]
    
    # Reward for keeping the pole upright (angle close to 0)
    pole_angle_temp = 1.0  # Temperature parameter for pole angle reward
    pole_angle_reward = torch.exp(-pole_angle_temp * torch.abs(pole_angle))
    
    # Reward for keeping the cart centered (position close to 0)
    cart_pos_temp = 1.0  # Temperature parameter for cart position reward
    cart_pos_reward = torch.exp(-cart_pos_temp * torch.abs(cart_pos))
    
    # Combine the rewards
    total_reward = pole_angle_reward + cart_pos_reward
    
    # Individual reward components for debugging
    reward_dict = {
        "pole_angle_reward": pole_angle_reward,
        "cart_pos_reward": cart_pos_reward
    }
    
    return total_reward, reward_dict
