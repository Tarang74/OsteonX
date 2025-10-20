"""Node density tests."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from osteonx.types import Osteon
from osteonx.generators import (
    spheres,
    cylinders,
    prisms,
    lattice_nodes_spherical,
    lattice_nodes_cylindrical,
    lattice_nodes_cartesian,
)
from osteonx.analysis import compute_edt, interpolate_surfaces, find_density
from osteonx.utils import scale_to_physical
from osteonx.visuals import plot_nodes_with_surface_2d, plot_nodes_3d

DOWNSCALING = (4, 4, 1)
SHAPE = (512 // DOWNSCALING[0], 512 // DOWNSCALING[1], 512 // DOWNSCALING[2])
UM_PER_VOXEL = (300.0, 300.0, 400.0)
CENTER = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)


def plot_results(t, counts, volumes, densities, filename):
    """Plot counts, volumes, and densities."""
    Path("./figures").mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Counts
    axs[0].plot(t[:-1], counts, "o-", label="Counts")
    axs[0].set_title("Counts")
    axs[0].set_xlabel("Normalised Time")
    axs[0].set_ylabel("Count")
    axs[0].legend()

    # Volumes
    axs[1].plot(t[:-1], volumes, "o-", label="Volumes")
    axs[1].set_title("Volumes")
    axs[1].set_xlabel("Normalised Time")
    axs[1].set_ylabel("Volume")
    axs[1].legend()

    # Densities
    axs[2].plot(t[:-1], densities, "o-", label="Densities")
    axs[2].set_title("Densities")
    axs[2].set_xlabel("Normalised Time")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"tests/figures/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def test_spheres():
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

    theta_samples = 20
    phi_samples = 15
    nodes = lattice_nodes_spherical(
        SHAPE,
        CENTER,
        outer_radius,
        inner_radius,
        r_samples=tsamples - 1,
        theta_samples=theta_samples,
        phi_samples=phi_samples,
    )

    counts, volumes = find_density(phi, nodes)
    volumes = scale_to_physical(volumes, osteon, dim=3)

    densities = counts / volumes

    # Compare with expected counts and volumes
    for i, (count, volume) in enumerate(zip(counts, volumes)):
        r_inner = (1 - t[i + 1]) * outer_radius + t[i + 1] * inner_radius
        r_outer = (1 - t[i]) * outer_radius + t[i] * inner_radius

        expected_count = theta_samples * phi_samples
        expected_volume = (
            (4 / 3 * np.pi * (r_outer**3 - r_inner**3))
            * np.prod(DOWNSCALING)
            * np.prod(UM_PER_VOXEL)
        )

        assert count == expected_count
        assert np.isclose(volume, expected_volume, rtol=0.1)

    plot_results(t, counts, volumes, densities, "test_sphere_nodes")
    plot_nodes_with_surface_2d(
        nodes, phi, layer=SHAPE[2] // 2, out="tests/figures/test_sphere_2d.png"
    )
    plot_nodes_3d(nodes, phi, out="tests/figures/test_sphere_3d.png", distance=200)


def test_cylinders():
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

    theta_samples = 20
    z_samples = 15
    nodes = lattice_nodes_cylindrical(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_radius,
        inner_radius,
        r_samples=tsamples - 1,
        theta_samples=theta_samples,
        z_samples=z_samples,
    )

    counts, volumes = find_density(phi, nodes)
    volumes = scale_to_physical(volumes, osteon, dim=3)

    densities = counts / volumes

    # Compare with expected counts and volumes
    for i, (count, volume) in enumerate(zip(counts, volumes)):
        r_inner = (1 - t[i + 1]) * outer_radius + t[i + 1] * inner_radius
        r_outer = (1 - t[i]) * outer_radius + t[i] * inner_radius

        expected_count = theta_samples * z_samples
        expected_volume = (
            (np.pi * (r_outer**2 - r_inner**2) * SHAPE[2])
            * np.prod(DOWNSCALING)
            * np.prod(UM_PER_VOXEL)
        )

        assert count == expected_count
        assert np.isclose(volume, expected_volume, rtol=0.1)

    plot_results(t, counts, volumes, densities, "test_cylinder_nodes")
    plot_nodes_with_surface_2d(
        nodes, phi, layer=0, out="tests/figures/test_cylinder_2d.png"
    )
    plot_nodes_3d(nodes, phi, out="tests/figures/test_cylinder_3d.png", distance=600)


def test_prisms():
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
    z_samples = 15
    nodes = lattice_nodes_cartesian(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_side,
        inner_side,
        x_samples=x_samples,
        y_samples=y_samples,
        z_samples=z_samples,
    )

    counts, volumes = find_density(phi, nodes)
    volumes = scale_to_physical(volumes, osteon, dim=3)

    densities = counts / volumes

    dx = (outer_side - inner_side) / x_samples
    dy = (outer_side - inner_side) / y_samples

    # Compare with expected counts and volumes
    for i, (count, volume) in enumerate(zip(counts, volumes)):
        side_inner = (1 - t[i + 1]) * outer_side + t[i + 1] * inner_side
        side_outer = (1 - t[i]) * outer_side + t[i] * inner_side

        x_outer = np.ceil(2 * side_outer / dx - 1)
        y_outer = np.ceil(2 * side_outer / dy - 1)
        x_inner = np.ceil(2 * side_inner / dx - 1)
        y_inner = np.ceil(2 * side_inner / dy - 1)
        expected_count = (x_outer * y_outer - x_inner * y_inner) * z_samples
        expected_volume = (
            4
            * (side_outer**2 - side_inner**2)
            * SHAPE[2]
            * np.prod(DOWNSCALING)
            * np.prod(UM_PER_VOXEL)
        )

        assert count == expected_count
        assert np.isclose(volume, expected_volume, rtol=0.1)

    plot_results(t, counts, volumes, densities, "test_prism_nodes")
    plot_nodes_with_surface_2d(
        nodes, phi, layer=0, out="tests/figures/test_prism_2d.png"
    )
    plot_nodes_3d(nodes, phi, out="tests/figures/test_prism_3d.png", distance=600)
