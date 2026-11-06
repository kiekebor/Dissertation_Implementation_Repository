
# ======================================================================
# MODEL PREDICTIVE CONTROL (MPC) — REUSABLE CONTROLLER FUNCTIONS
# ======================================================================

import numpy as np

# Reuse the RK45 dynamics integrator developed for CatheterEnv.
from .dynamics import integrate_catheter


# Add this import alongside the existing imports at the top of mpc.py.
from dataclasses import dataclass

# Import our existing nearest-centreline calculation.
from .geometry import centreline_distance

# Import the SciPy numerical optimiser for the MPC control problem.
from scipy.optimize import minimize

# Import the performance counter for timing optimisations.
from time import perf_counter 

# ======================================================================
# 1. PREDICT FUTURE CATHETER STATES
# ======================================================================

def predict_trajectory(initial_state, control_sequence, vessel, env_config):
    """
    Predict catheter motion over a sequence of control forces using RK45.

    Parameters:
        initial_state: Array-like, shape (4,).
            Initial physical catheter state [x, y, vx, vy].
        control_sequence: Array-like, shape (N, 2).
            Proposed control forces [Fx, Fy] at each prediction step (N).
        vessel: SVessel
            Vessel geometry containing centreline points, arc lengths,
            and local vessel radii.
        env_config: EnvironmentConfig
            Physical parameters, simulation timestep, and RK45 tolerances.

    Returns:
        trajectory: ndarray, shape (N + 1, 4).
            Predicted catheter states, including the initial state.
    """

    # Convert inputs to numerical arrays for consistent calculations.
    initial_state = np.asarray(initial_state, dtype=float)
    control_sequence = np.asarray(control_sequence, dtype=float)

    # Check that the initial state contains [x, y, vx, vy].
    if initial_state.shape != (4,):
        raise ValueError("initial_state must have shape (4,).")

    # Each control action must contain two force components [Fx, Fy].
    if control_sequence.ndim != 2 or control_sequence.shape[1] != 2:
        raise ValueError("control_sequence must have shape (N, 2).")

    # Determine the number of timesteps in the prediction horizon.
    horizon = control_sequence.shape[0]

    # Allocate storage for the initial state and all future states.
    trajectory = np.zeros((horizon + 1, 4), dtype=float)

    # Store the initial catheter state as the first trajectory entry.
    trajectory[0] = initial_state

    # Predict the catheter state at each successive timestep.
    for k in range(horizon):

        # Use the previous predicted state as the starting condition.
        current_state = trajectory[k]

        # Retrieve the proposed force for the current prediction step.
        control_force = control_sequence[k]

        # Integrate the catheter dynamics over one simulation timestep.
        next_state = integrate_catheter(
            state=current_state,
            control_force=control_force,
            dt=env_config.dt,
            centreline=vessel.centreline,
            s=vessel.s,
            local_radius=vessel.local_radius,
            max_flow_speed=env_config.max_flow_speed,
            mass=env_config.mass,
            blood_density=env_config.blood_density,
            drag_coefficient=env_config.drag_coefficient,
            reference_area=env_config.reference_area,
            rtol=env_config.rtol,
            atol=env_config.atol
        )

        # Save the predicted state for use in the next timestep.
        trajectory[k + 1] = next_state

    # Return the complete predicted state sequence.
    return trajectory


# ======================================================================
# 2. MPC COST FUNCTION CONFIGURATION
# ======================================================================

@dataclass
class MPCCostConfig:
    """Configurable weights and wall-penalty settings for MPC."""

    progress_weight: float = 1.0
    control_weight: float = 0.01
    wall_weight: float = 10.0
    terminal_weight: float = 5.0
    safety_margin_fraction: float = 0.5
    exit_penalty: float = 100.0

# ======================================================================
# 3. CONTINUOUS CENTRELINE PROGRESS FOR MPC OPTIMISATION
# ======================================================================

def continuous_centreline_progress(position, vessel):
    """
    Calculate continuous remaining centreline distance to the target.

    The catheter position is projected onto the nearest line segment
    connecting consecutive centreline samples. Progress is interpolated
    along that segment rather than restricted to sampled points.

    Parameters:
        position: array-like, shape (2,).
            Current catheter position [x, y] in metres.
        vessel: SVessel
            Vessel containing centreline coordinates, cumulative
            arc lengths, and the target location.

    Returns:
        d_target: float
            Signed remaining centreline distance to the target (m).
            Positive before the target, zero at the target, and
            negative after passing the target.
    """

    # Convert the catheter position into a NumPy array.
    position = np.asarray(position, dtype=float)

    # Retrieve the sampled centreline and cumulative arc lengths.
    centreline = vessel.centreline
    s = vessel.s

    # Define every centreline segment using consecutive sample points.
    segment_starts = centreline[:-1]
    segment_ends = centreline[1:]

    # Calculate the direction vector of each centreline segment.
    segment_vectors = segment_ends - segment_starts

    # Calculate the squared length of each segment.
    segment_lengths_squared = np.sum(segment_vectors ** 2, axis=1)

    # Calculate vectors from each segment start to the catheter position.
    relative_vectors = position - segment_starts

    # Calculate the fractional projection along each segment.
    # A value of 0 corresponds to the start and 1 to the end.
    projection_fractions = np.sum(
        relative_vectors * segment_vectors, axis=1
    ) / segment_lengths_squared

    # Restrict each projection to its finite line segment.
    projection_fractions = np.clip(projection_fractions, 0.0, 1.0)

    # Calculate the projected point on every centreline segment.
    projected_points = (
        segment_starts
        + projection_fractions[:, None] * segment_vectors
    )

    # Calculate squared distances from the catheter to each projection.
    distances_squared = np.sum(
        (projected_points - position) ** 2, axis=1
    )

    # Identify the segment with the closest projected point.
    nearest_segment = int(np.argmin(distances_squared))

    # Retrieve the projection fraction on the selected segment.
    fraction = projection_fractions[nearest_segment]

    # Interpolate cumulative centreline distance along the segment.
    projected_s = (
        s[nearest_segment]
        + fraction * (s[nearest_segment + 1] - s[nearest_segment])
    )

    # Obtain the target's arc-length coordinate using the existing
    # distance-to-target function at the start of the centreline.
    # The cumulative arc length at centreline[0] is zero.
    target_s = vessel.distance_to_target(vessel.centreline[0])

    # Calculate remaining signed progress relative to the target.
    d_target = target_s - projected_s

    return float(d_target)


# ======================================================================
# 4. CALCULATE MPC TRAJECTORY COST
# ======================================================================

def calculate_mpc_cost(
    trajectory,
    control_sequence,
    vessel,
    env_config,
    cost_config
):
    """
    Calculate the finite-horizon MPC objective for a predicted trajectory.

    Parameters:
        trajectory: ndarray, shape (N + 1, 4).
            Predicted physical states [x, y, vx, vy].
        control_sequence: ndarray, shape (N, 2).
            Proposed forces [Fx, Fy] in Newtons.
        vessel: SVessel
            Vessel geometry and target information.
        env_config: EnvironmentConfig
            Environment settings, including maximum control force.
        cost_config: MPCCostConfig
            Weights and wall-proximity penalty parameters.

    Returns:
        total_cost: float
            Normalised weighted MPC objective. Lower values are preferred.
    """

    # Convert the supplied trajectory and forces to NumPy arrays.
    trajectory = np.asarray(trajectory, dtype=float)
    control_sequence = np.asarray(control_sequence, dtype=float)

    # Number of actions in the prediction horizon.
    horizon = len(control_sequence)

    # Check that the supplied trajectory matches the action sequence.
    if control_sequence.shape != (horizon, 2):
        raise ValueError("control_sequence must have shape (N, 2).")

    if trajectory.shape != (horizon + 1, 4):
        raise ValueError("trajectory must have shape (N + 1, 4).")

    # The safety margin must be positive to allow normalisation.
    if not 0.0 < cost_config.safety_margin_fraction <= 1.0:
        raise ValueError("safety_margin_fraction must be in (0, 1].")

    if cost_config.exit_penalty < 0.0:
        raise ValueError("exit_penalty must be non-negative.")

    # Initialise the accumulated MPC cost.
    total_cost = 0.0

    # Evaluate the cost at each prediction step, excluding the terminal state.
    for k in range(horizon):

        # Obtain the catheter position at this prediction step.
        position = trajectory[k, :2]

        # Calculate continuous remaining centreline distance to the target.
        # Segment projection avoids staircase changes in measured progress.
        d_target = continuous_centreline_progress(position, vessel)

        # Normalise the target distance using the total vessel length.
        progress_cost = (d_target / vessel.length) ** 2

        # Normalise the two applied control-force components.
        normalised_force = control_sequence[k] / env_config.max_force

        # Penalise squared control-force magnitude.
        control_cost = np.sum(normalised_force ** 2)

        # Calculate signed wall clearance: positive inside, negative outside.
        d_wall, inside_vessel = vessel.wall_clearance(position)

        # Find the local vessel radius at the nearest centreline point.
        _, _, nearest_index = centreline_distance(
            position, vessel.centreline
        )
        local_radius = vessel.local_radius[nearest_index]

        # Define the safety margin as a fraction of the local radius.
        d_safe = cost_config.safety_margin_fraction * local_radius

        # Express signed wall clearance relative to the safety margin.
        normalised_clearance = d_wall / d_safe

        # Penalise proximity to the wall, including boundary crossings.
        proximity_penalty = max(0.0, 1.0 - normalised_clearance) ** 2
        violation_penalty = (
            cost_config.exit_penalty
            * max(0.0, -normalised_clearance) ** 2
        )

        wall_cost = proximity_penalty + violation_penalty

        # Add the three weighted running-cost components.
        total_cost += (
            cost_config.progress_weight * progress_cost
            + cost_config.control_weight * control_cost
            + cost_config.wall_weight * wall_cost
        )

    # Calculate the terminal target error using continuous centreline progress.
    # Small changes in the predicted endpoint can now affect the terminal cost.
    terminal_position = trajectory[-1, :2]
    terminal_distance = continuous_centreline_progress(
        terminal_position, vessel
    )

    # Penalise remaining target distance at the prediction-horizon endpoint.
    terminal_cost = (terminal_distance / vessel.length) ** 2

    total_cost += cost_config.terminal_weight * terminal_cost

    return float(total_cost)


# ======================================================================
# 5. OPTIMISE THE MPC CONTROL SEQUENCE USING SCIPY
# ======================================================================

def optimise_control_sequence(
    initial_state,
    vessel,
    env_config,
    cost_config,
    horizon=5,
    initial_guess=None,
    maxiter=50,
    ftol=1e-5
):
    """
    Optimise a finite sequence of catheter control forces using SciPy SLSQP.

    Parameters:
        initial_state: ndarray, shape (4,).
            Current catheter state [x, y, vx, vy].
        vessel: SVessel
            Vessel geometry used for prediction and cost evaluation.
        env_config: EnvironmentConfig
            Physical parameters and maximum control force.
        cost_config: MPCCostConfig
            Configurable MPC objective weights and wall-penalty settings.
        horizon: int
            Number of future control steps to optimise.
        initial_guess: ndarray, shape (horizon, 2), optional
            Initial sequence of physical control forces in Newtons.
            If None, initialise all proposed forces to zero.
        maxiter: int
            Maximum number of SLSQP optimisation iterations.
        ftol: float
            Solver convergence tolerance for the optimisation objective.

    Returns:
        optimal_sequence: ndarray, shape (horizon, 2).
            Best control-force sequence returned by the optimiser, in N.
        result: scipy.optimize.OptimizeResult
            Optimisation diagnostics, including success status, objective
            value, iteration count and solver termination message.
    """

    # Validate the prediction horizon.
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    # Validate the initial physical state.
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (4,):
        raise ValueError("initial_state must have shape (4,).")

    # Ensure that the maximum allowable control force is positive.
    max_force = env_config.max_force

    if max_force <= 0.0:
        raise ValueError("max_force must be positive.")

    # Initialise the proposed force sequence.
    if initial_guess is None:
        initial_guess = np.zeros((horizon, 2), dtype=float)
    else:
        initial_guess = np.asarray(initial_guess, dtype=float)

        if initial_guess.shape != (horizon, 2):
            raise ValueError(
                "initial_guess must have shape (horizon, 2)."
            )

    # Scale physical forces in Newtons into normalised values [-1, 1].
    # This improves numerical conditioning for the SciPy optimiser.
    initial_guess_normalised = np.clip(
        initial_guess / max_force,
        -1.0,
        1.0
    )

    # SciPy minimise() expects a flat one-dimensional decision vector.
    x0 = initial_guess_normalised.flatten()

    # Define the objective evaluated repeatedly by SciPy.
    def objective(normalised_controls):

        # Convert the flat decision vector into N two-component actions.
        normalised_sequence = normalised_controls.reshape(horizon, 2)

        # Convert normalised actions back into physical forces (Newtons).
        control_sequence = normalised_sequence * max_force

        # Predict the resulting catheter trajectory using RK45.
        trajectory = predict_trajectory(
            initial_state=initial_state,
            control_sequence=control_sequence,
            vessel=vessel,
            env_config=env_config
        )

        # Calculate the MPC cost for the predicted trajectory.
        total_cost = calculate_mpc_cost(
            trajectory=trajectory,
            control_sequence=control_sequence,
            vessel=vessel,
            env_config=env_config,
            cost_config=cost_config
        )

        return total_cost

    # Each normalised force component is constrained to [-1, 1].
    # There are two force components for every prediction step.
    bounds = [(-1.0, 1.0)] * (2 * horizon)

    # Run the Sequential Least Squares Programming optimiser.
    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": maxiter,   # Maximum optimisation iterations.
            "ftol": ftol,         # Optimisation convergence tolerance.
            "eps": 1e-3,         # Finite-difference step in normalised units.
            "disp": False        # Suppress solver messages during execution.
        }
    )

    # Convert the returned normalised solution into physical forces.
    optimal_sequence = (
        np.clip(result.x, -1.0, 1.0).reshape(horizon, 2)
        * max_force
    )

    # Return both the force sequence and optimisation diagnostics.
    return optimal_sequence, result



# ======================================================================
# 6. CLOSED-LOOP MPC CONTROLLER WITH WARM-STARTING
# ======================================================================

class MPCController:
    """
    Receding-horizon MPC controller for catheter navigation.

    At each control step, the controller optimises a sequence of future
    forces, applies the first force, and stores the remaining sequence
    as the initial guess for the next optimisation.

    Parameters:
        vessel: SVessel
            Vessel geometry used for MPC prediction and cost evaluation.
        env_config: EnvironmentConfig
            Physical model parameters and maximum control force.
        cost_config: MPCCostConfig
            MPC objective weights and wall-penalty settings.
        horizon: int
            Number of future control steps in the prediction horizon.
        maxiter: int
            Maximum number of SLSQP optimisation iterations.
        ftol: float
            SLSQP convergence tolerance.

    Attributes:
        initial_guess: ndarray, shape (horizon, 2).
            Control sequence used to initialise the next optimisation.
        optimisation_history: list
            Solver diagnostics recorded at each control step.
    """

    def __init__(
        self,
        vessel,
        env_config,
        cost_config,
        horizon=5,
        maxiter=50,
        ftol=1e-5
    ):
        # Store the vessel and physical parameters.
        self.vessel = vessel
        self.env_config = env_config

        # Store MPC cost weights and numerical optimisation settings.
        self.cost_config = cost_config
        self.horizon = horizon
        self.maxiter = maxiter
        self.ftol = ftol

        # Initialise the controller's internal state.
        self.reset()

    def reset(self):
        """
        Reset the warm-start sequence and optimisation diagnostics.

        Call this method before beginning a new navigation episode.
        """

        # Start the next optimisation from zero control forces.
        self.initial_guess = np.zeros(
            (self.horizon, 2), dtype=float
        )

        # Clear diagnostics from any previous navigation episode.
        self.optimisation_history = []

    def __call__(self, observation):
        """
        Calculate the next MPC control force.

        Parameters:
            observation: array-like, shape (6,).
                Environment observation [x, y, vx, vy,
                target_progress, wall_clearance].

        Returns:
            action: ndarray, shape (2,).
                Control force [Fx, Fy] in Newtons.
        """

        # Convert the environment observation to a numerical array.
        observation = np.asarray(observation, dtype=float)

        # Validate the expected observation dimensions.
        if observation.shape != (6,):
            raise ValueError("observation must have shape (6,).")

        # MPC requires the four-dimensional physical catheter state.
        # The final two observation components are not needed here.
        current_state = observation[:4]

        # Measure the computation time for this control decision.
        start_time = perf_counter()

        # Optimise future forces using the previous solution as a warm start.
        optimal_sequence, result = optimise_control_sequence(
            initial_state=current_state,
            vessel=self.vessel,
            env_config=self.env_config,
            cost_config=self.cost_config,
            horizon=self.horizon,
            initial_guess=self.initial_guess,
            maxiter=self.maxiter,
            ftol=self.ftol
        )

        # Record elapsed wall-clock optimisation time.
        elapsed_time = perf_counter() - start_time

        # Check whether the solver returned a valid numerical sequence.
        valid_solution = (
            result.success
            and np.all(np.isfinite(optimal_sequence))
        )

        if valid_solution:

            # Apply only the first force of the optimal sequence.
            action = optimal_sequence[0].copy()

            # Shift the remaining sequence forward by one timestep.
            # Repeat the last action to preserve the horizon length.
            self.initial_guess = np.vstack([
                optimal_sequence[1:],
                optimal_sequence[-1:]
            ])

        else:

            # If optimisation fails, fall back to the first force
            # from the previous feasible initial-guess sequence.
            action = self.initial_guess[0].copy()

            # Shift the fallback sequence for the following timestep.
            self.initial_guess = np.vstack([
                self.initial_guess[1:],
                self.initial_guess[-1:]
            ])

        # Ensure the selected action respects physical force limits.
        action = np.clip(
            action,
            -self.env_config.max_force,
            self.env_config.max_force
        )

        # Save diagnostics for later computational-performance analysis.
        self.optimisation_history.append({
            "success": bool(result.success),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "optimisation_time": float(elapsed_time),
            "objective_value": float(result.fun),
            "message": str(result.message)
        })

        # Return the force in the format expected by CatheterEnv.
        return action.astype(np.float32)
