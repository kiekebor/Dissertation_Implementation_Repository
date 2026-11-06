
# ======================================================================
# CATHETER DYNAMICS AND NUMERICAL INTEGRATION
# ======================================================================

import numpy as np
from scipy.integrate import solve_ivp

from .geometry import local_directions, centreline_distance


# ============================================================
# BLOOD-FLOW VELOCITY
# ============================================================

# Here, using the local tangent (local blood flow direction) and lateral offset, we calculate the local Poiseuille-inspired blood-flow velocity.
# The flow speed is maximum at the centreline and decreases parabolically to zero at the vessel walls.

def blood_flow_velocity(position, centreline, s, local_radius, max_flow_speed):
    """
    Calculate the local Poiseuille-inspired blood-flow velocity.

    Parameters:
        position: Catheter position [x, y] (m).
        centreline: Sampled centreline coordinates (m).
        s: Cumulative centreline distances (m).
        local_radius: Vessel half-width at each centreline point (m).
        max_flow_speed: Maximum blood-flow speed (m/s).

    Returns:
        flow_velocity: Two-dimensional blood-flow velocity [vx, vy] (m/s).
    """

    # Obtain the local flow direction and lateral offset.
    tangent, _, signed_offset = local_directions(
        position, centreline, s
    )

    # Identify the nearest centreline location.
    _, _, nearest_index = centreline_distance(
        position, centreline
    )

    # Get the vessel radius at that location.
    radius = local_radius[nearest_index]

    # Calculate blood-flow speed using the parabolic profile.
    flow_speed = max_flow_speed * (
        1 - (signed_offset / radius)**2
    )

    # Prevent negative flow speeds outside the nominal profile.
    flow_speed = max(0.0, flow_speed)

    # Convert scalar speed to a two-dimensional velocity vector.
    flow_velocity = flow_speed * tangent

    return flow_velocity


# ============================================================
# CATHETER DYNAMICS WITH QUADRATIC DRAG
# ============================================================

#With this function, we calculate the catheter's acceleration based on Newton's second law, considering both the control force and the quadratic drag force from the blood flow.

def catheter_dynamics(state, control_force, flow_velocity,
                      mass, blood_density, drag_coefficient, reference_area):
    """
    Calculate catheter state derivatives using Newton's second law.

    Parameters:
        state: [x, y, vx, vy] (m, m/s).
        control_force: Applied force [Fx, Fy] (N).
        flow_velocity: Local blood velocity [ux, uy] (m/s).
        mass: Effective catheter tip mass (kg).
        blood_density: Blood density (kg/m^3).
        drag_coefficient: Dimensionless drag coefficient.
        reference_area: Catheter reference area (m^2).

    Returns:
        state_derivative: [vx, vy, ax, ay].
    """

    # Check physical parameter validity.
    if mass <= 0 or blood_density <= 0 or reference_area <= 0:
        raise ValueError("Mass, blood density and reference area must be positive.")

    if drag_coefficient < 0:
        raise ValueError("Drag coefficient cannot be negative.")

    # Extract catheter velocity from the state.
    catheter_velocity = state[2:4]

    # Calculate blood velocity relative to the catheter.
    relative_velocity = flow_velocity - catheter_velocity

    # Calculate quadratic drag force from the proposal.
    drag_force = (
        0.5 * blood_density * drag_coefficient * reference_area
        * np.linalg.norm(relative_velocity)
        * relative_velocity
    )

    # Calculate catheter acceleration.
    acceleration = (control_force + drag_force) / mass

    # Construct the state derivative.
    state_derivative = np.concatenate(
        (catheter_velocity, acceleration)
    )

    return state_derivative


# ============================================================
# CATHETER STATE UPDATE USING RK45
# ============================================================

#Here, we integrate the catheter's state over a single timestep using the RK45 method.
#The Scipy `solve_ivp` function is used to perform the integration, which allows for adaptive step sizing and error control.

def integrate_catheter(state, control_force, dt, centreline, s,
                       local_radius, max_flow_speed, mass,
                       blood_density, drag_coefficient, reference_area,
                       rtol=1e-6, atol=1e-9):
    """
    Advance the catheter state by one timestep using RK45.

    Parameters:
        state: Current catheter state [x, y, vx, vy].
        control_force: Applied force [Fx, Fy] (N).
        dt: Simulation timestep (s).
        centreline, s, local_radius: Vessel geometry parameters for the centreline, cumulative distance, and local radius.
        max_flow_speed: Maximum blood-flow speed (m/s).
        mass: Effective catheter mass (kg).
        blood_density: Blood density (kg/m^3).
        drag_coefficient: Dimensionless drag coefficient.
        reference_area: Catheter reference area (m^2).
        rtol, atol: RK45 error tolerances.

    Returns:
        next_state: Updated catheter state [x, y, vx, vy].
    """

    if dt <= 0:
        raise ValueError("Timestep must be positive.")

    # Define the differential equations evaluated by RK45.
    def combined_dynamics(t, current_state):

        # Obtain the catheter's current estimated position.
        position = current_state[:2]

        # Recalculate blood velocity at that position.
        flow_velocity = blood_flow_velocity(
            position, centreline, s, local_radius, max_flow_speed
        )

        # Calculate the corresponding state derivatives.
        return catheter_dynamics(
            current_state, control_force, flow_velocity,
            mass, blood_density, drag_coefficient, reference_area
        )

    # Integrate over one simulation timestep.
    solution = solve_ivp(
        combined_dynamics,
        t_span=(0.0, dt),
        y0=state,
        method="RK45",
        rtol=rtol,
        atol=atol
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # Extract the state at the end of the timestep.
    next_state = solution.y[:, -1]

    return next_state
