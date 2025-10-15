"""Edge/segment surface density tests."""

import numpy as np

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


DOWNSAMPLE = 4
SHAPE = (512 // DOWNSAMPLE, 512 // DOWNSAMPLE, 512 // DOWNSAMPLE)
UM_PER_VOXEL = (300.0, 300.0, 400.0)
CENTER = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)


def test_sphere_edges():
    """Test sphere with edge/segment surface density analysis."""
    outer_radius_voxels = 200 // DOWNSAMPLE
    inner_radius_voxels = 80 // DOWNSAMPLE
    tsamples = 5

    outer, inner = spheres(SHAPE, CENTER, outer_radius_voxels, inner_radius_voxels)
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=(DOWNSAMPLE, DOWNSAMPLE, DOWNSAMPLE),
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    segments, deltas = lattice_polylines_spherical(
        SHAPE,
        CENTER,
        outer_radius_voxels,
        inner_radius_voxels,
        r_samples=tsamples,
        theta_samples=20,
        phi_samples=15,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)


def test_cylinder_edges():
    """Test cylinder with edge/segment surface density analysis."""
    outer_radius_voxels = 200 // DOWNSAMPLE
    inner_radius_voxels = 80 // DOWNSAMPLE
    tsamples = 5

    outer, inner = cylinders(
        SHAPE, (CENTER[0], CENTER[1]), outer_radius_voxels, inner_radius_voxels
    )
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=(DOWNSAMPLE, DOWNSAMPLE, DOWNSAMPLE),
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    segments, deltas = lattice_polylines_cylindrical(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_radius_voxels,
        inner_radius_voxels,
        r_samples=tsamples,
        theta_samples=24,
        z_samples=12,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)


def test_prism_edges():
    """Test prism with edge/segment surface density analysis."""
    outer_side_voxels = 360 // DOWNSAMPLE
    inner_side_voxels = 200 // DOWNSAMPLE
    tsamples = 5

    outer, inner = prisms(
        SHAPE, (CENTER[0], CENTER[1]), outer_side_voxels, inner_side_voxels
    )
    osteon = Osteon(
        outer=outer,
        inner=inner,
        shape=SHAPE,
        um_per_voxel=UM_PER_VOXEL,
        downsample=(DOWNSAMPLE, DOWNSAMPLE, DOWNSAMPLE),
    )

    dts = compute_edt(osteon, save_array=False)
    t, phi = interpolate_surfaces(osteon, dts, tsamples=tsamples)

    segments, deltas = lattice_polylines_cartesian(
        SHAPE,
        (CENTER[0], CENTER[1]),
        outer_side_voxels,
        inner_side_voxels,
        r_samples=tsamples,
        z_samples=10,
    )

    counts, areas = find_surface_density(phi, osteon, segments, deltas)
