
# ======================================================================
# CONTROLLER-INDEPENDENT EVALUATION OF CATHETER NAVIGATION
# ======================================================================

import numpy as np
from collections import Counter


# ======================================================================
# 1. EVALUATE A SINGLE NAVIGATION EPISODE
# ======================================================================


def evaluate_episode(
    env,
    controller,
    seed=None,
    record_trajectory=True,
    reset_options=None
):
    """
    Evaluate a controller over one complete catheter navigation episode.

    Parameters:
        env: Gymnasium catheter navigation environment.
        controller: Callable receiving an observation and returning
                    a two-dimensional control force [Fx, Fy].
        seed: Optional random seed for reproducibility.
        record_trajectory: Whether to store physical states and actions.
        reset_options: Optional dictionary specifying the initial position
                       and velocity passed to the environment reset method.

    Returns:
        result: Dictionary containing episode outcome, performance
                measurements, and optional trajectory/action histories.
    """

    # Reset the environment using the specified starting conditions.
    observation, info = env.reset(seed=seed, options=reset_options)

    total_reward = 0.0
    terminated = False
    truncated = False

    # Record the initial physical state if requested.
    states = [env.state.copy()] if record_trajectory else None
    actions = [] if record_trajectory else None

    # Continue until success, vessel exit, or time-limit truncation.
    while not (terminated or truncated):

        # Obtain the control action from SAC, MPC, or another controller.
        action = np.asarray(controller(observation), dtype=np.float32)

        # Advance the environment by one timestep.
        observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate the environment reward.
        total_reward += reward

        # Store the physical state and applied control force.
        if record_trajectory:
            states.append(env.state.copy())
            actions.append(action.copy())

    # Identify the final navigation outcome.
    success = bool(info["is_success"])
    outcome = info["outcome"]

    # Completion time is defined only for successful episodes.
    completion_time = env.steps * env.config.dt if success else None

    result = {
        "success": success,
        "outcome": outcome,
        "steps": env.steps,
        "completion_time": completion_time,
        "total_reward": float(total_reward),
        "wall_violation": outcome == "side_wall_violation",
        "trajectory": np.asarray(states) if record_trajectory else None,
        "actions": np.asarray(actions) if record_trajectory else None
    }

    return result



# ======================================================================
# 2. SUMMARISE PERFORMANCE ACROSS MULTIPLE EPISODES
# ======================================================================

def summarise_evaluations(episode_results):
    """
    Summarise navigation performance across multiple evaluation episodes.

    Parameters:
        episode_results: List of dictionaries returned by evaluate_episode().

    Returns:
        summary: Dictionary containing success rate, wall violation rate,
                 completion-time statistics, episode steps and outcome counts.
    """

    if len(episode_results) == 0:
        raise ValueError("At least one episode result is required.")

    # Count episodes and their navigation outcomes.
    n_episodes = len(episode_results)
    n_success = sum(result["success"] for result in episode_results)
    n_wall_violations = sum(
        result["wall_violation"] for result in episode_results
    )

    # Calculate the two primary percentage-based performance metrics.
    success_rate = 100.0 * n_success / n_episodes
    wall_violation_rate = 100.0 * n_wall_violations / n_episodes

    # Collect completion times from successful episodes only.
    completion_times = np.array([
        result["completion_time"]
        for result in episode_results
        if result["success"]
    ], dtype=float)

    # Completion time is undefined if no episodes succeed.
    if len(completion_times) > 0:
        mean_completion_time = float(np.mean(completion_times))
        completion_time_std = float(np.std(completion_times))
    else:
        mean_completion_time = float("nan")
        completion_time_std = float("nan")

    # Calculate the average episode length, including unsuccessful episodes.
    mean_episode_steps = float(np.mean([
        result["steps"] for result in episode_results
    ]))

    # Count each termination outcome for further analysis.
    outcome_counts = dict(Counter(
        result["outcome"] for result in episode_results
    ))

    summary = {
        "n_episodes": n_episodes,
        "success_rate": success_rate,
        "wall_violation_rate": wall_violation_rate,
        "mean_completion_time": mean_completion_time,
        "completion_time_std": completion_time_std,
        "mean_episode_steps": mean_episode_steps,
        "outcome_counts": outcome_counts
    }

    return summary
