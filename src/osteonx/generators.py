from typing import Tuple

import numpy as np


def spheres(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    R_outer: float,
    R_inner: float,
):
    """Generate spherical shell masks.

    Args:
        shape: (width, height, depth) grid dimensions.
        center: (x0, y0, z0) center coordinates.
        R_outer: Outer radius.
        R_inner: Inner radius.

    Returns:
        Tuple[ndarray, ndarray]: (outer, inner) boolean masks.

    Raises:
        ValueError: if R_outer is too large for the shape.
    """
    if R_outer > min(shape) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    x0, y0, z0 = center

    outer = np.full(shape, False, dtype=bool)
    inner = np.full(shape, False, dtype=bool)

    for z in range(shape[2]):
        dist = (X - x0) ** 2 + (Y - y0) ** 2 + (z - z0) ** 2
        outer[:, :, z] = dist < R_outer**2
        inner[:, :, z] = dist >= R_inner**2

    return outer, inner


def cylinders(
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    R_outer: float,
    R_inner: float,
):
    """Generate cylindrical shell masks.

    Args:
        shape: (width, height, depth) grid dimensions.
        center: (x0, y0) center in the x-y plane.
        R_outer: Outer radius.
        R_inner: Inner radius.

    Returns:
        Tuple[ndarray, ndarray]: (outer, inner) boolean masks.

    Raises:
        ValueError: if R_outer is too large for the shape.
    """
    if R_outer > min(shape[0], shape[1]) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    x0, y0 = center

    dist = (X - x0) ** 2 + (Y - y0) ** 2

    inner = np.repeat((dist >= R_inner**2)[:, :, np.newaxis], shape[2], axis=2)
    outer = np.repeat((dist < R_outer**2)[:, :, np.newaxis], shape[2], axis=2)

    return outer, inner


def prisms(
    shape: Tuple[int, int, int], center: Tuple[int, int], S_outer: int, S_inner: int
):
    """Generate square prism shell masks.

    Args:
        shape: (width, height, depth) grid dimensions.
        center: (x0, y0) center in the x-y plane.
        S_outer: Outer side length.
        S_inner: Inner side length.

    Returns:
        Tuple[ndarray, ndarray]: (outer, inner) boolean masks.
    """
    x0, y0 = center

    outer = np.full(shape, False, dtype=bool)
    inner = np.full(shape, True, dtype=bool)

    outer[x0 - S_outer : x0 + S_outer, y0 - S_outer : y0 + S_outer, :] = True

    inner[x0 - S_inner : x0 + S_inner, y0 - S_inner : y0 + S_inner, :] = False

    return outer, inner


def lattice_nodes_spherical(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    R_outer: float,
    R_inner: float,
    r_samples: int,
    theta_samples: int,
    phi_samples: int,
) -> np.ndarray:
    """Generate lattice nodes in spherical coordinates.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0, z0) center coordinates.
        R_outer: Outer radius.
        R_inner: Inner radius.
        r_samples: Number of radial samples.
        theta_samples: Number of azimuthal samples.
        phi_samples: Number of polar samples.

    Returns:
        ndarray: (N, 3) nodes as [x, y, z].
    """
    if R_outer > min(shape) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    x0, y0, z0 = center

    dr = (R_outer - R_inner) / r_samples
    r = np.linspace(R_inner, R_outer, r_samples, endpoint=False) + dr / 2
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    phi = np.linspace(0, np.pi, phi_samples, endpoint=False)

    R, THETA, PHI = np.meshgrid(r, theta, phi, indexing="ij")

    X = x0 + R * np.sin(PHI) * np.cos(THETA)
    Y = y0 + R * np.sin(PHI) * np.sin(THETA)
    Z = z0 + R * np.cos(PHI)

    return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)


def lattice_nodes_cylindrical(
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    R_outer: float,
    R_inner: float,
    r_samples: int,
    theta_samples: int,
    z_samples: int,
) -> np.ndarray:
    """Generate lattice nodes in cylindrical coordinates.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        R_outer: Outer radius.
        R_inner: Inner radius.
        r_samples: Number of radial samples.
        theta_samples: Number of angular samples.
        z_samples: Number of z samples.

    Returns:
        ndarray: (N, 3) nodes as [x, y, z].
    """
    if R_outer > min(shape[0], shape[1]) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    x0, y0 = center
    z_layers = shape[2]

    dr = (R_outer - R_inner) / r_samples
    r = np.linspace(R_inner, R_outer, r_samples, endpoint=False) + dr / 2
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    z = np.linspace(0, z_layers, z_samples, endpoint=False)

    R, THETA, Z = np.meshgrid(r, theta, z, indexing="ij")

    X = x0 + R * np.cos(THETA)
    Y = y0 + R * np.sin(THETA)

    return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)


def lattice_nodes_cartesian(
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    S_outer: float,
    S_inner: float,
    x_samples: int,
    y_samples: int,
    z_samples: int,
) -> np.ndarray:
    """Generate lattice nodes in Cartesian coordinates.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        S_outer: Outer side length.
        S_inner: Inner side length.
        x_samples: Number of x samples.
        y_samples: Number of y samples.
        z_samples: Number of z samples.

    Returns:
        ndarray: (N, 3) nodes as [x, y, z].
    """
    dx = (S_outer - S_inner) / x_samples
    dy = (S_outer - S_inner) / y_samples

    x = center[0] + np.arange(-S_outer, S_outer - dx, dx) + dx / 2
    y = center[1] + np.arange(-S_outer, S_outer - dy, dy) + dy / 2
    z = np.linspace(0, shape[2], z_samples, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    triplets = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    filter = ~(
        np.logical_and(
            (center[0] - S_inner) <= triplets[:, 0],
            triplets[:, 0] <= (center[0] + S_inner),
        )
        & np.logical_and(
            center[1] - S_inner <= triplets[:, 1],
            triplets[:, 1] <= (center[1] + S_inner),
        )
    )
    return triplets[filter]


def lattice_polylines_spherical(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    R_outer: float,
    R_inner: float,
    r_samples: int,
    theta_samples: int,
    phi_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate radial polylines from spherical lattice nodes.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0, z0) center coordinates.
        R_outer: Outer radius.
        R_inner: Inner radius.
        r_samples: Number of radial samples.
        theta_samples: Number of azimuthal samples.
        phi_samples: Number of polar samples.

    Returns:
        Tuple[ndarray, ndarray]: (segments, deltas) arrays.
    """
    if R_outer > min(shape) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    x0, y0, z0 = center

    dr = (R_outer - R_inner) / r_samples
    r = np.linspace(R_inner, R_outer, r_samples, endpoint=False) + dr / 2
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    phi = np.linspace(0, np.pi, phi_samples, endpoint=False)

    segments = []
    deltas = []

    for th in theta:
        for ph in phi:
            for ri in range(r_samples - 1):
                r1 = r[ri]
                r2 = r[ri + 1]

                x1 = x0 + r1 * np.sin(ph) * np.cos(th)
                y1 = y0 + r1 * np.sin(ph) * np.sin(th)
                z1 = z0 + r1 * np.cos(ph)

                x2 = x0 + r2 * np.sin(ph) * np.cos(th)
                y2 = y0 + r2 * np.sin(ph) * np.sin(th)
                z2 = z0 + r2 * np.cos(ph)

                segments.append([x1, y1, z1])
                deltas.append([x2 - x1, y2 - y1, z2 - z1])

    return np.array(segments), np.array(deltas)


def lattice_polylines_cylindrical(
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    R_outer: float,
    R_inner: float,
    r_samples: int,
    theta_samples: int,
    z_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate radial polylines from cylindrical lattice nodes.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        R_outer: Outer radius.
        R_inner: Inner radius.
        r_samples: Number of radial samples.
        theta_samples: Number of angular samples.
        z_samples: Number of z samples.

    Returns:
        Tuple[ndarray, ndarray]: (segments, deltas) arrays.
    """
    if R_outer > min(shape) / 2:
        raise ValueError("R_outer exceeds half the smallest shape dimension.")

    x0, y0 = center
    z_layers = shape[2]

    dr = (R_outer - R_inner) / r_samples
    r = np.linspace(R_inner, R_outer, r_samples, endpoint=False) + dr / 2
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    z = np.linspace(0, z_layers, z_samples, endpoint=False)

    segments = []
    deltas = []

    for th in theta:
        for z_val in z:
            for ri in range(r_samples - 1):
                r1 = r[ri]
                r2 = r[ri + 1]

                x1 = x0 + r1 * np.cos(th)
                y1 = y0 + r1 * np.sin(th)

                x2 = x0 + r2 * np.cos(th)
                y2 = y0 + r2 * np.sin(th)

                segments.append([x1, y1, z_val])
                deltas.append([x2 - x1, y2 - y1, 0])

    return np.array(segments), np.array(deltas)


def lattice_polylines_cartesian(
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    S_outer: float,
    S_inner: float,
    x_samples: int,
    y_samples: int,
    z_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cardinal polylines for a square prism.

    Args:
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        S_outer: Outer side length.
        S_inner: Inner side length.
        x_samples: Number of x samples.
        y_samples: Number of y samples.
        z_samples: Number of z samples.

    Returns:
        Tuple[ndarray, ndarray]: (segments, deltas) arrays.
    """
    dx = (S_outer - S_inner) / x_samples
    dy = (S_outer - S_inner) / y_samples

    z = np.linspace(0, shape[2], z_samples, endpoint=False)

    segments = []
    deltas = []

    for xi in range(x_samples - 1):
        x_pos = center[0] + S_inner + xi * dx + dx / 2
        x_neg = center[0] - S_inner - xi * dx - dx / 2

        for yi in range(y_samples - 1):
            y_pos = center[1] + S_inner + yi * dy + dy / 2
            y_neg = center[1] - S_inner - yi * dy - dy / 2

            for zc in z:
                # Positive x direction
                segments.append([x_pos, center[1] + yi * dy + dy / 2, zc])
                deltas.append([dx, 0, 0])
                segments.append([x_pos, center[1] - yi * dy - dy / 2, zc])
                deltas.append([dx, 0, 0])

                # Negative x direction
                segments.append([x_neg, center[1] + yi * dy + dy / 2, zc])
                deltas.append([-dx, 0, 0])
                segments.append([x_neg, center[1] - yi * dy - dy / 2, zc])
                deltas.append([-dx, 0, 0])

                # Positive y direction
                segments.append([center[0] + xi * dx + dx / 2, y_pos, zc])
                deltas.append([0, dy, 0])
                segments.append([center[0] - xi * dx - dx / 2, y_pos, zc])
                deltas.append([0, dy, 0])

                # Negative y direction
                segments.append([center[0] + xi * dx + dx / 2, y_neg, zc])
                deltas.append([0, -dy, 0])
                segments.append([center[0] - xi * dx - dx / 2, y_neg, zc])
                deltas.append([0, -dy, 0])

    return np.array(segments), np.array(deltas)


def random_nodes_cylindrical(
    n_nodes: int,
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    R_inner: float,
    R_outer: float,
) -> np.ndarray:
    """
    Generate nodes uniformly distributed within a cylindrical mask.

    Args:
        n_nodes: Number of nodes to generate.
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        R_inner: Inner radius of the mask.
        R_outer: Outer radius of the mask.

    Returns:
        ndarray: (n_nodes, 3) array of node coordinates.
    """

    angle = np.random.uniform(0, 2 * np.pi, n_nodes)
    r = np.sqrt(np.random.uniform(R_inner**2, R_outer**2, n_nodes))
    z = np.random.uniform(0, shape[2], n_nodes)
    x0, y0 = center
    x = x0 + r * np.cos(angle)
    y = y0 + r * np.sin(angle)
    return np.stack([x, y, z], axis=1)


def random_nodes_spherical(
    n_nodes: int,
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    R_inner: float,
    R_outer: float,
) -> np.ndarray:
    """
    Generate nodes uniformly distributed within a spherical mask.

    Args:
        n_nodes: Number of nodes to generate.
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0, z0) center coordinates.
        R_inner: Inner radius of the mask.
        R_outer: Outer radius of the mask.

    Returns:
        ndarray: (n_nodes, 3) array of node coordinates.
    """
    x0, y0, z0 = center
    r = np.cbrt(np.random.uniform(R_inner**3, R_outer**3, n_nodes))
    theta = np.random.uniform(0, 2 * np.pi, n_nodes)
    phi = np.random.uniform(0, np.pi, n_nodes)

    x = x0 + r * np.sin(phi) * np.cos(theta)
    y = y0 + r * np.sin(phi) * np.sin(theta)
    z = z0 + r * np.cos(phi)

    return np.stack([x, y, z], axis=1)


def random_nodes_cartesian(
    n_nodes: int,
    shape: Tuple[int, int, int],
    center: Tuple[float, float],
    S_inner: float,
    S_outer: float,
) -> np.ndarray:
    """
    Generate nodes uniformly distributed within a Cartesian prismatic shell.

    Args:
        n_nodes: Number of nodes to generate.
        shape: Grid dimensions (width, height, depth).
        center: (x0, y0) center in the x-y plane.
        S_inner: Inner side length of the exclusion box.
        S_outer: Outer side length of the bounding box.

    Returns:
        ndarray: (n_nodes, 3) array of node coordinates.
    """
    x0, y0 = center

    # Calculate bounds based on geometry parameters
    x_min, x_max = x0 - S_outer / 2, x0 + S_outer / 2
    y_min, y_max = y0 - S_outer / 2, y0 + S_outer / 2
    z_min, z_max = 0, shape[2]

    x_ex_min, x_ex_max = x0 - S_inner / 2, x0 + S_inner / 2
    y_ex_min, y_ex_max = y0 - S_inner / 2, y0 + S_inner / 2

    nodes = []
    while len(nodes) < n_nodes:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)

        if not (x_ex_min <= x <= x_ex_max and y_ex_min <= y <= y_ex_max):
            nodes.append([x, y, z])

    return np.array(nodes)
