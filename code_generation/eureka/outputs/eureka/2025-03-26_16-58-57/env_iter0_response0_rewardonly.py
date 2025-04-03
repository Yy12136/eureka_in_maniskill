@torch.jit.script
def compute_reward(
    dof_pos: torch.Tensor,  # Joint positions [num_envs, 2] (cart position, pole angle)
    dof_vel: torch.Tensor,  # Joint velocities [num_envs, 2] (cart velocity, pole angular velocity)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract cart position and pole angle
    cart_pos = dof_pos[:, 0]  # Cart position
    pole_angle = dof_pos[:, 1]  # Pole angle

    # Extract cart velocity and pole angular velocity
    cart_vel = dof_vel[:, 0]  # Cart velocity
    pole_vel = dof_vel[:, 1]  # Pole angular velocity

    # Reward for keeping the pole upright (angle close to 0)
    angle_temp = 10.0  # Temperature for angle reward
    angle_reward = torch.exp(-angle_temp * torch.abs(pole_angle))

    # Reward for keeping the cart centered (position close to 0)
    pos_temp = 1.0  # Temperature for position reward
    pos_reward = torch.exp(-pos_temp * torch.abs(cart_pos))

    # Penalize high velocities to encourage smooth control
    vel_temp = 0.1  # Temperature for velocity penalty
    vel_penalty = torch.exp(-vel_temp * (torch.abs(cart_vel) + torch.abs(pole_vel)))

    # Combine rewards and penalties
    total_reward = angle_reward * pos_reward * vel_penalty

    # Individual reward components for debugging
    reward_dict = {
        "angle_reward": angle_reward,
        "pos_reward": pos_reward,
        "vel_penalty": vel_penalty,
    }

    return total_reward, reward_dict
