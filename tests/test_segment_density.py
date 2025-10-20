"""Edge/segment surface density tests."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from osteonx.types import Osteon
from osteonx.generators import (
    spheres,
    cylinders,
    prisms,
    lattice_polylines_spherical,
    lattice_polylines_cylindrical,
    lattice_polylines_cartesian,
)
from osteonx.analysis import compute_edt, interpolate_surfaces, find_surface_density
from osteonx.visuals import plot_segments_with_surface_2d, plot_segments_3d

DOWNSCALING = (4, 4, 1)
SHAPE = (512 // DOWNSCALING[0], 512 // DOWNSCALING[1], 512 // DOWNSCALING[2])
UM_PER_VOXEL = (300.0, 300.0, 400.0)
CENTER = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)


def ellipsoid_surface_area(a, b, c, p=1.6075):
    return 4 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)


def ellipse_perimeter(a, b):
    eccentricity = np.sqrt(1 - (b**2 / a**2))
    epsilon = 3 * a * eccentricity**20 / 2**36
    return np.pi * (
        (a + b)
        + 3 * (a - b) ** 2 / (10 * (a + b) + np.sqrt(a**2 + 14 * a * b + b**2))
        + epsilon
    )


def plot_results(t, counts, areas, densities, filename):
    """Plot counts, areas, and densities."""
    Path("./figures").mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Counts
    axs[0].plot(t[1:-1], counts, "o-", label="Counts")
    axs[0].set_title("Counts")
    axs[0].set_xlabel("Normalised Time")
    axs[0].set_ylabel("Count")
    axs[0].legend()

    # Areas
    axs[1].plot(t[1:-1], areas, "o-", label="Areas")
    axs[1].set_title("Areas")
    axs[1].set_xlabel("Normalised Time")
    axs[1].set_ylabel("Area")
    axs[1].legend()

    # Densities
    axs[2].plot(t[1:-1], densities, "o-", label="Densities")
    axs[2].set_title("Densities")
    axs[2].set_xlabel("Normalised Time")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"tests/figures/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def test_sphere_edges():
    """Test sphere with edge/segment surface density analysis."""
    outer_radius = 200 // DOWNSCALING[0]
    inner_radius = 80 // DOWNSCALING[0]
    tsamples = 5

    outer, inner = spheres(SHAPE, CENTER, outer_radius, inner_radius)
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=DOWNSCALING,
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    segments, deltas = lattice_polylines_spherical(
        SHAPE,
        CENTER,
        outer_radius,
        inner_radius,
        r_samples=tsamples - 1,
        theta_samples=20,
        phi_samples=15,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)
    densities = counts / areas

    # Compare with expected counts and areas
    for i, (count, area) in enumerate(zip(counts, areas)):
        r = (1 - t[i + 1]) * outer_radius + t[i + 1] * inner_radius

        a, b, c = (
            r * UM_PER_VOXEL[0],
            r * UM_PER_VOXEL[1],
            r * UM_PER_VOXEL[2],
        )
        expected_count = 20 * 15
        expected_area = ellipsoid_surface_area(a, b, c)

        assert count == expected_count
        assert np.isclose(area, expected_area, rtol=0.1)

    plot_results(t, counts, areas, densities, "test_sphere_segments")
    plot_segments_with_surface_2d(
        segments,
        deltas,
        phi,
        layer=SHAPE[2] // 2,
        out="tests/figures/test_sphere_segments_2d.png",
    )
    plot_segments_3d(
        segments, deltas, phi, out="tests/figures/test_sphere_segments_3d.png", distance=200
    )


def test_cylinder_edges():
    """Test cylinder with edge/segment surface density analysis."""
    outer_radius = 200 // DOWNSCALING[0]
    inner_radius = 80 // DOWNSCALING[0]
    tsamples = 5

    outer, inner = cylinders(SHAPE, (CENTER[0], CENTER[1]), outer_radius, inner_radius)
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=DOWNSCALING,
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    segments, deltas = lattice_polylines_cylindrical(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_radius,
        inner_radius,
        r_samples=tsamples - 1,
        theta_samples=24,
        z_samples=12,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)
    densities = counts / areas

    # Compare with expected counts and areas
    for i, (count, area) in enumerate(zip(counts, areas)):
        r = (1 - t[i + 1]) * outer_radius + t[i + 1] * inner_radius

        a, b = (
            r * UM_PER_VOXEL[0],
            r * UM_PER_VOXEL[1],
        )
        h = SHAPE[2] * DOWNSCALING[2] * UM_PER_VOXEL[2]
        expected_count = 24 * 12
        expected_area = ellipse_perimeter(a, b) * h

        assert count == expected_count
        assert np.isclose(area, expected_area, rtol=0.1)

    plot_results(t, counts, areas, densities, "test_cylinder_segments")
    plot_segments_with_surface_2d(
        segments, deltas, phi, layer=0, out="tests/figures/test_cylinder_segments_2d.png"
    )
    plot_segments_3d(
        segments, deltas, phi, out="tests/figures/test_cylinder_segments_3d.png", distance=600
    )


def test_prism_edges():
    """Test prism with edge/segment surface density analysis."""
    outer_side = 200 // DOWNSCALING[0]
    inner_side = 80 // DOWNSCALING[0]
    tsamples = 5

    outer, inner = prisms(SHAPE, (CENTER[0], CENTER[1]), outer_side, inner_side)
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=DOWNSCALING,
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    x_samples = tsamples - 1
    y_samples = tsamples - 1
    z_samples = 10

    segments, deltas = lattice_polylines_cartesian(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_side,
        inner_side,
        x_samples=x_samples,
        y_samples=y_samples,
        z_samples=z_samples,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)
    densities = counts / areas

    # Compare with expected counts and areas
    for i, (count, area) in enumerate(zip(counts, areas)):
        S = (1 - t[i + 1]) * outer_side + t[i + 1] * inner_side

        expected_count = 4 * ((x_samples - 1) + (y_samples - 1)) * z_samples
        expected_area = 2 * 4 * (S * SHAPE[2]) * np.prod(UM_PER_VOXEL)

        assert count == expected_count
        assert np.isclose(area, expected_area, rtol=1) # large tolerance as surfaces are not perfect square prisms

    plot_results(t, counts, areas, densities, "test_prism_segments")
    plot_segments_with_surface_2d(
        segments, deltas, phi, layer=0, out="tests/figures/test_prism_segments_2d.png"
    )
    plot_segments_3d(
        segments, deltas, phi, out="tests/figures/test_prism_segments_3d.png", distance=600
    )
