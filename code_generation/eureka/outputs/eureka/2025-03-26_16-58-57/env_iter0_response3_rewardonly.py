@torch.jit.script
def compute_reward(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the pole angle (dof_pos[:, 1]) and cart velocity (dof_vel[:, 0])
    pole_angle = dof_pos[:, 1]
    cart_velocity = dof_vel[:, 0]

    # Reward for keeping the pole upright (minimize the angle from the vertical position)
    angle_temperature = 1.0
    angle_reward = torch.exp(-angle_temperature * torch.abs(pole_angle))

    # Reward for keeping the cart stable (minimize the cart's velocity)
    velocity_temperature = 0.1
    velocity_reward = torch.exp(-velocity_temperature * torch.abs(cart_velocity))

    # Total reward is a combination of angle_reward and velocity_reward
    total_reward = angle_reward + velocity_reward

    # Return the total reward and individual reward components
    reward_components = {
        "angle_reward": angle_reward,
        "velocity_reward": velocity_reward
    }
    return total_reward, reward_components
