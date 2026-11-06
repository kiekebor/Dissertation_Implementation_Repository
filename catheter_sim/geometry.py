
# ======================================================================
# S-SHAPED VESSEL GEOMETRY AND GEOMETRIC CALCULATIONS
# ======================================================================

import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path

from .config import SVesselConfig


# ======================================================================
# 1. GENERATE THE S-SHAPED VESSEL CENTRELINE
# ======================================================================

def generate_s_centreline(config):
    """
    Generate two tangentially connected circular arcs.

    Parameters:
        config: SVesselConfig containing the arc parameters.

    Returns:
        centreline: Array of [x, y] coordinates (m).
        join_index: Index where the two arcs connect.
    """

    def generate_arc(start, heading, radius, angle, points):
        """Generate an arc from its start point and direction."""

        theta = np.linspace(0, angle, points)

        # Calculate positions along the circular arc.
        x = start[0] + radius * (
            np.sin(heading + theta) - np.sin(heading)
        )

        y = start[1] - radius * (
            np.cos(heading + theta) - np.cos(heading)
        )

        return np.column_stack((x, y))

    # Generate the first arc (anticlockwise).
    start = np.array([config.start_x, config.start_y])

    arc1 = generate_arc(
        start,
        config.start_angle,
        config.arc_radius_1,
        config.arc_angle_1,
        config.num_points
    )

    # The second arc begins where the first arc ends.
    join_point = arc1[-1]

    # Tangent direction at the end of the first arc.
    join_heading = config.start_angle + config.arc_angle_1

    # Generate the second arc in the opposite direction.
    # Negating the angle reverses the curvature.
    arc2 = generate_arc(
        join_point,
        join_heading,
        -config.arc_radius_2,
        -config.arc_angle_2,
        config.num_points
    )

    # Combine both arcs without duplicating the connection point.
    centreline = np.vstack((arc1, arc2[1:]))

    join_index = len(arc1) - 1

    return centreline, join_index


# ======================================================================
# 2. DEFINE VESSEL WIDTH PROFILES
# ======================================================================

def constant_width(s, radius):
    """
    Return a constant vessel half-width along the centreline.

    Parameters:
        s: Distances along the vessel centreline (m).
        radius: Constant vessel half-width (m).

    Returns:
        Array of vessel half-widths (m).
    """
    return np.full_like(s, radius, dtype=float)


def smoothstep(t):
    """
    Calculate smoothstep interpolation for values between 0 and 1.

    h(t) = 3t² - 2t³

    The function has zero slope at both endpoints.
    """
    return 3*t**2 - 2*t**3


def variable_width(s, radius, centre_fraction=0.60,
                   stenosis_length=0.020, severity=0.30):
    """
    Generate a smooth, symmetric stenosis along the vessel centreline.

    Parameters
    ----------
    s : np.ndarray
        Cumulative distance along the vessel centreline (m).
    radius : float
        Nominal vessel radius (m).
    centre_fraction : float
        Location of stenosis centre as a fraction of vessel length.
    stenosis_length : float
        Total length of the stenosis (m).
    severity : float
        Maximum fractional reduction in vessel radius.

    Returns
    -------
    np.ndarray
        Local vessel radius at each centreline position (m).
    """

    # Validate the stenosis parameters.
    if not 0 <= centre_fraction <= 1:
        raise ValueError("centre_fraction must be between 0 and 1.")
    if not 0 < severity < 1:
        raise ValueError("severity must be between 0 and 1.")
    if radius <= 0 or stenosis_length <= 0:
        raise ValueError("Radius and stenosis length must be positive.")

    # Determine the stenosis centre, start and end along the centreline.
    vessel_length = s[-1]
    stenosis_centre = centre_fraction * vessel_length
    stenosis_start = stenosis_centre - stenosis_length / 2
    stenosis_end = stenosis_centre + stenosis_length / 2

    # Ensure the entire stenosis fits inside the vessel.
    if stenosis_start < 0 or stenosis_end > vessel_length:
        raise ValueError("Stenosis extends beyond the vessel boundaries.")

    # Initialise the radius profile at the nominal radius.
    local_radius = np.full_like(s, radius, dtype=float)

    # Identify the contraction and expansion sections.
    contraction = (s >= stenosis_start) & (s <= stenosis_centre)
    expansion = (s > stenosis_centre) & (s <= stenosis_end)

    # Convert distances in each half into transition fractions [0, 1].
    t_left = (s[contraction] - stenosis_start) / (stenosis_length / 2)
    t_right = (s[expansion] - stenosis_centre) / (stenosis_length / 2)

    # Decrease the radius smoothly during contraction.
    local_radius[contraction] = radius * (
        1 - severity * smoothstep(t_left)
    )

    # Restore the radius smoothly during expansion.
    local_radius[expansion] = radius * (
        1 - severity * smoothstep(1 - t_right)
    )

    return local_radius


# ======================================================================
# 3. GENERATE VESSEL WALLS
# ======================================================================

def generate_vessel_walls(centreline, width_profile, radius):
    """
    Generate vessel walls perpendicular to the centreline.

    Parameters:
        centreline: Array of [x, y] coordinates (m).
        width_profile: Function returning local half-widths.
        radius: Baseline vessel half-width (m).

    Returns:
        wall1, wall2: Vessel boundary coordinates (m).
        s: Cumulative distance along the centreline (m).
    """

    # Calculate cumulative distance along the centreline.
    segments = np.diff(centreline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    # Estimate the local tangent direction.
    dx = np.gradient(centreline[:, 0], s)
    dy = np.gradient(centreline[:, 1], s)

    tangents = np.column_stack((dx, dy))
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    # Rotate each tangent by 90 degrees to obtain its unit normal.
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    # Obtain the local half-width from the supplied profile.
    local_radius = width_profile(s, radius)

    # Generate both walls using the local normal directions.
    wall1 = centreline + local_radius[:, None] * normals
    wall2 = centreline - local_radius[:, None] * normals

    return wall1, wall2, s


# ======================================================================
# 4. CENTRELINE DISTANCE AND WALL CLEARANCE
# ======================================================================

def centreline_distance(position, centreline):
    """
    Find the nearest sampled centreline point.
    This is a simplified approximation to determine the catheter's distance from the vessel centreline.

    Parameters:
        position: Catheter position [x, y] (m).
        centreline: Array of centreline coordinates (m).

    Returns:
        distance: Distance to the nearest centreline point (m).
        nearest_point: Coordinates of the nearest point (m).
        nearest_index: Index of that point in the centreline.
    """

    # Build a spatial search tree from the centreline coordinates.
    tree = cKDTree(centreline)

    # Find the nearest centreline point to the catheter.
    distance, nearest_index = tree.query(position)

    nearest_point = centreline[nearest_index]

    return distance, nearest_point, nearest_index


def wall_clearance(position, wall1, wall2):
    """
    Calculate the distance to the nearest sampled vessel wall point.

    Parameters:
        position: Catheter position [x, y] (m).
        wall1: First vessel wall coordinates (m).
        wall2: Second vessel wall coordinates (m).

    Returns:
        distance: Distance to the nearest wall point (m).
        nearest_point: Coordinates of the nearest wall point (m).
    """

    # Combine both vessel walls into one array.
    wall_points = np.vstack((wall1, wall2))

    # Build a search tree and find the nearest wall point.
    tree = cKDTree(wall_points)
    distance, nearest_index = tree.query(position)

    nearest_point = wall_points[nearest_index]

    return distance, nearest_point


def signed_wall_clearance(position, wall1, wall2, tolerance=1e-8):
    """
    Calculate signed wall clearance for a catheter position.

    Parameters:
        position: Catheter position [x, y] (m).
        wall1: First vessel wall coordinates (m).
        wall2: Second vessel wall coordinates (m).
        tolerance: Tolerance for boundary contact (m).

    Returns:
        signed_distance: Positive inside, negative outside (m).
        inside: Boolean indicating whether the catheter is inside.
    """

    # Combine both walls into a closed vessel boundary.
    boundary = np.vstack((wall1, wall2[::-1]))

    # Create a polygon representing the vessel interior.
    vessel_polygon = Path(boundary, closed=True)

    # Check whether the catheter lies inside the vessel.
    inside = bool(vessel_polygon.contains_point(position))

    # Calculate the nearest-wall distance.
    distance, _ = wall_clearance(position, wall1, wall2)

    # Treat very small distances as boundary contact.
    if distance <= tolerance:
        signed_distance = 0.0
    elif inside:
        signed_distance = distance
    else:
        signed_distance = -distance

    return signed_distance, inside


# ======================================================================
# 5. TARGET LOCATION AND REMAINING CENTRELINE DISTANCE
# ======================================================================

def define_target(centreline, s, target_fraction):
    """
    Define a target along the sampled vessel centreline.

    parameters:
        centreline: Array of [x, y] coordinates (m).
        s: Cumulative distance along the centreline (m).
        target_fraction: Fraction of total centreline length for the target.

    Returns:
        target_position: Target coordinates [x, y] (m).
        target_index: Index of the nearest centreline point.
    """

    # Ensure the target fraction is within the permitted range.
    if not 0 < target_fraction < 1:
        raise ValueError("target_fraction must be between 0 and 1.")

    # Calculate the desired target distance along the vessel.
    target_s = target_fraction * s[-1]

    # Find the sampled point closest to the desired distance.
    target_index = np.argmin(np.abs(s - target_s))

    # Retrieve the corresponding target coordinates.
    target_position = centreline[target_index]

    return target_position, target_index


def distance_to_target(position, centreline, s, target_index):
    """
    Calculate remaining centreline distance to a specified target.

    parameters:
        position: Catheter position [x, y] (m).
        centreline: Array of [x, y] coordinates (m).
        s: Cumulative distance along the centreline (m).
        target_index: Index of the target point on the centreline.

    Returns:
        remaining_distance: Signed centreline distance to target (m).
        nearest_index: Nearest centreline index to catheter.
    """

    # Locate the catheter along the sampled centreline.
    _, _, nearest_index = centreline_distance(
        position, centreline
    )

    # Get cumulative distances of catheter and target.
    current_s = s[nearest_index]
    target_s = s[target_index]

    # Negative remaining distance indicates passing the target.
    remaining_distance = target_s - current_s

    return remaining_distance, nearest_index


# ======================================================================
# 6. LOCAL TANGENT AND NORMAL DIRECTIONS
# ======================================================================

def local_directions(position, centreline, s):
    """
    Calculate the local unit tangent and normal vectors.

    parameters:
        position: Catheter position [x, y] (m).
        centreline: Array of [x, y] coordinates (m).
        s: Cumulative distance along the centreline (m).

    Returns:
        tangent: Unit vector along the centreline.
        normal: Unit vector perpendicular to the centreline.
        signed_offset: Signed catheter deviation (m).
    """

    # Find the nearest sampled centreline point.
    _, nearest_point, nearest_index = centreline_distance(
        position, centreline
    )

    # Estimate the tangent vectors along the centreline.
    dx = np.gradient(centreline[:, 0], s)
    dy = np.gradient(centreline[:, 1], s)

    # Select the tangent at the nearest centreline point.
    tangent = np.array([
        dx[nearest_index],
        dy[nearest_index]
    ])

    # Normalise the tangent to unit length.
    tangent = tangent / np.linalg.norm(tangent)

    # Rotate the tangent by 90 degrees to get the normal.
    normal = np.array([-tangent[1], tangent[0]])

    # Calculate signed displacement from the centreline.
    displacement = position - nearest_point
    signed_offset = np.dot(displacement, normal)

    return tangent, normal, signed_offset


# ======================================================================
# 7. S-SHAPED VESSEL GEOMETRY CLASS
# ======================================================================

class SVessel:
    """Store S-shaped vessel geometry and provide geometric calculations."""

    def __init__(self, config, width_profile, radius, target_fraction):

        # Generate the centreline using the previously defined function.
        self.centreline, self.join_index = generate_s_centreline(config)

        # Generate vessel walls and cumulative centreline distances.
        self.wall1, self.wall2, self.s = generate_vessel_walls(
            self.centreline, width_profile, radius
        )

        # Store the local vessel radius at each centreline point.
        self.local_radius = width_profile(self.s, radius)

        # Validate vessel radii.
        if not np.all(np.isfinite(self.local_radius)) or np.any(self.local_radius <= 0):
            raise ValueError("All vessel radii must be finite and positive.")

        # Define the navigation target.
        self.target_position, self.target_index = define_target(
            self.centreline, self.s, target_fraction
        )

        # Store the total centreline length.
        self.length = self.s[-1]

    def distance_to_target(self, position):
        """Return signed remaining centreline progress to the target."""

        remaining_distance, _ = distance_to_target(
            position, self.centreline, self.s, self.target_index
        )

        return remaining_distance

    def wall_clearance(self, position):
        """Return signed wall clearance and inside/outside classification."""

        return signed_wall_clearance(
            position, self.wall1, self.wall2
        )

    def flow_direction(self, position):
        """Return the local unit tangent vector."""

        tangent, _, _ = local_directions(
            position, self.centreline, self.s
        )

        return tangent

    def lateral_offset(self, position):
        """Return the signed lateral offset from the centreline."""

        _, _, signed_offset = local_directions(
            position, self.centreline, self.s
        )

        return signed_offset

    def classify_boundary_crossing(self, previous_position, new_position):
        """
        Identify the first vessel boundary crossed during a catheter movement.

        The movement is approximated by a straight line between positions.
        Boundary intersections are checked against both side walls,
        the entrance and the distal end.

        Returns:
            "side_wall_violation", "entrance_exit", "distal_exit",
            or None if no crossing is detected.
        """

        p = np.asarray(previous_position, dtype=float)
        r = np.asarray(new_position, dtype=float) - p

        # Validate the catheter positions.
        if p.shape != (2,) or r.shape != (2,):
            raise ValueError("Catheter positions must have two coordinates.")

        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError("Catheter positions must contain finite values.")

        if np.all(r == 0):
            return None

        # Construct segments for both side walls, entrance and distal end.
        boundary_start = np.vstack((
            self.wall1[:-1],
            self.wall2[:-1],
            self.wall1[0],
            self.wall1[-1]
        ))

        boundary_end = np.vstack((
            self.wall1[1:],
            self.wall2[1:],
            self.wall2[0],
            self.wall2[-1]
        ))

        # Assign an exit category to each boundary segment.
        boundary_types = (
            ["side_wall_violation"] * (2 * (len(self.wall1) - 1))
            + ["entrance_exit", "distal_exit"]
        )

        # Represent each boundary as a starting point and direction.
        q = boundary_start
        s = boundary_end - boundary_start
        difference = q - p

        def cross2d(a, b):
            """Calculate the two-dimensional cross product."""
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        # Calculate the intersection parameters for each boundary segment.
        denominator = cross2d(r, s)
        parallel = np.abs(denominator) < 1e-14

        # Avoid division by zero for parallel segments.
        safe_denominator = np.where(parallel, 1.0, denominator)

        t = cross2d(difference, s) / safe_denominator
        u = cross2d(difference, r) / safe_denominator

        # A valid intersection must lie within both line segments.
        intersects = (
            ~parallel
            & (t >= 0.0) & (t <= 1.0)
            & (u >= 0.0) & (u <= 1.0)
        )

        if not np.any(intersects):
            return None

        # Select the first boundary encountered along the movement.
        first_index = np.argmin(np.where(intersects, t, np.inf))

        return boundary_types[first_index]
