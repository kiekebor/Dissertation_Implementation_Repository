
# ======================================================================
# CONFIGURATION CLASSES FOR THE CATHETER NAVIGATION SIMULATION
# ======================================================================

from dataclasses import dataclass
import numpy as np


@dataclass
class SVesselConfig:
    """Configurable parameters for the S-shaped centreline."""

    # Radii of curvature for the two circular arcs (m).
    arc_radius_1: float
    arc_radius_2: float

    # Angular extent of each arc (radians).
    arc_angle_1: float
    arc_angle_2: float

    # Starting position of the centreline (m).
    start_x: float
    start_y: float

    # Initial centreline direction (radians).
    start_angle: float

    # Number of points used to sample each arc.
    num_points: int


@dataclass
class EnvironmentConfig:
    """Configurable physical, numerical and navigation parameters."""

    # Catheter and blood-flow parameters.
    mass: float
    blood_density: float
    drag_coefficient: float
    reference_area: float
    max_flow_speed: float

    # Controller force limits.
    max_force: float

    # Numerical integration settings.
    dt: float
    rtol: float
    atol: float

    # Navigation task settings.
    target_tolerance: float
    max_episode_steps: int

    def __post_init__(self):
        """Validate configuration parameters after initialisation."""

        positive_values = {
            "mass": self.mass,
            "blood_density": self.blood_density,
            "reference_area": self.reference_area,
            "max_force": self.max_force,
            "dt": self.dt,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_episode_steps": self.max_episode_steps
        }

        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        nonnegative_values = {
            "drag_coefficient": self.drag_coefficient,
            "max_flow_speed": self.max_flow_speed,
            "target_tolerance": self.target_tolerance
        }

        for name, value in nonnegative_values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

        if not isinstance(self.max_episode_steps, int):
            raise ValueError("max_episode_steps must be an integer.")


@dataclass
class RewardConfig:
    """Configurable weights and terminal rewards for SAC training."""

    progress_weight: float
    wall_weight: float
    control_weight: float
    success_bonus: float
    failure_penalty: float