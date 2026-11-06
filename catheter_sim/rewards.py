
# ======================================================================
# NAVIGATION REWARD FUNCTION
# ======================================================================

import numpy as np

from .config import RewardConfig


def calculate_reward(previous_d_target, current_d_target, d_wall,
                     local_radius, control_force, vessel_length,
                     max_force, reward_config, success, failure):
    """
    Calculate the navigation reward for one simulation step.

    Parameters:
        previous_d_target, current_d_target: Signed target progress (m).
        d_wall: Signed nearest-wall clearance (m).
        local_radius: Vessel half-width at the catheter location (m).
        control_force: Applied controller force [Fx, Fy] (N).
        vessel_length: Total centreline length (m).
        max_force: Maximum force per action component (N).
        reward_config: Reward weights and terminal values.
        success, failure: Boolean episode outcomes.

    Returns:
        reward: Total scalar reward.
        components: Dictionary containing individual reward terms.
    """

    # Reward forward progress along the vessel.
    progress_reward = reward_config.progress_weight * (
        (previous_d_target - current_d_target) / vessel_length
    )

    # Penalise proximity to the vessel wall.
    wall_penalty = np.clip(
        1.0 - d_wall / local_radius, 0.0, 1.0
    )
    wall_reward = -reward_config.wall_weight * wall_penalty

    # Penalise unnecessarily large control forces.
    normalised_force = control_force / max_force
    control_reward = -reward_config.control_weight * np.sum(
        normalised_force**2
    )

    # Apply terminal rewards or penalties.
    terminal_reward = (
        reward_config.success_bonus * success
        - reward_config.failure_penalty * failure
    )

    reward = (
        progress_reward + wall_reward
        + control_reward + terminal_reward
    )

    components = {
        "progress": float(progress_reward),
        "wall": float(wall_reward),
        "control": float(control_reward),
        "terminal": float(terminal_reward)
    }

    return float(reward), components