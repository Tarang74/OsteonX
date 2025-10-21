"""
Derivative Methods Comparison Example

This script demonstrates and compares different derivative strategies for density
profiles in osteon structures using synthetic cell data. It includes:

1. Exact derivative of smoothing spline
2. Backward difference
3. Central difference
4. Forward difference

The density is calculated as:
    ρ(t) = N'(t) / V'(t)
where N(t) is the cumulative cell count and V(t) is the cumulative volume.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from osteonx import analysis, generators, types, utils

plt.ioff()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare different smoothing methods for density profiles"
    )
    parser.add_argument(
        "--ncells", type=int, default=100000, help="Number of cells (default: 100000)"
    )
    parser.add_argument(
        "--tsamples",
        type=int,
        default=20,
        help="Number of interpolated layers (default: 20)",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="figures",
        help="Output directory for figures (default: figures)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    figures_output_path = Path(args.figure_dir)
    figures_output_path.mkdir(parents=True, exist_ok=True)

    shape = (512, 512, 100)
    downsampling = (2, 2, 1)
    center = (shape[0] // 2, shape[1] // 2)
    R_outer = 200 // downsampling[0]
    R_inner = 50 // downsampling[0]
    um_per_voxel = (1.0, 1.0, 1.0)

    # Generate synthetic data
    cells = generators.random_cells_cylindrical(
        args.ncells, shape, center, R_inner, R_outer
    )
    outer, inner = generators.cylinders(shape, center, R_outer, R_inner)
    osteon = types.Osteon(
        shape=shape,
        outer=outer,
        inner=inner,
        um_per_voxel=um_per_voxel,
        downsample=downsampling,
    )

    # Compute EDTs
    dts = analysis.compute_edt(osteon, save_array=False)

    # Interpolate surfaces
    t, phi = analysis.interpolate_surfaces(osteon, dts, tsamples=args.tsamples)
    dt = t[1] - t[0]

    # Count cells and measure volumes
    counts, volumes = analysis.find_density(phi, cells)
    volumes_physical = utils.scale_to_physical(volumes, osteon)

    # Calculate density from counts and volumes
    raw_density = counts / volumes_physical
    avg_density = np.sum(counts) / np.sum(volumes_physical)

    # Calculate density from cumulative counts and volumes
    cumulative_counts = np.cumsum(np.hstack([0, counts]))
    cumulative_volumes_physical = np.cumsum(np.hstack([0, volumes_physical]))

    # Derivative methods

    # 1. Exact derivative of smoothing spline
    t2, (s_cumulative_counts, s_cumulative_volumes) = utils.smooth(
        t,
        (cumulative_counts, cumulative_volumes_physical),
        lam=0.001,
        upsample=5,
    )
    s_density_cumulative = s_cumulative_counts.derivative()(
        t2
    ) / s_cumulative_volumes.derivative()(t2)

    # 2. Backward difference
    fb_counts = utils.difference(cumulative_counts, method="backward", prepend=0)
    fb_volumes = utils.difference(
        cumulative_volumes_physical, method="backward", prepend=0
    )
    fb_density = fb_counts / fb_volumes

    # 3. Central difference
    fc_counts = utils.difference(cumulative_counts, method="central")
    fc_volumes = utils.difference(cumulative_volumes_physical, method="central")
    fc_density = fc_counts / fc_volumes

    # 4. Forward difference
    ff_counts = utils.difference(
        cumulative_counts, method="forward", append=cumulative_counts[-1]
    )
    ff_volumes = utils.difference(
        cumulative_volumes_physical,
        method="forward",
        append=cumulative_volumes_physical[-1],
    )
    ff_density = ff_counts / ff_volumes

    # Plot counts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(t[:-1], counts, "o", markersize=4, label="Raw counts")
    ax1.plot(
        t2,
        s_cumulative_counts.derivative()(t2) * dt,
        "r-",
        linewidth=2,
        label="$s'(N(t))$",
    )
    ax1.plot(t[:-1] + dt / 2, fb_counts, "g--", linewidth=2, label="$N'_b(t)$")
    ax1.plot(t, fc_counts, "m-.", linewidth=2, label="$N'_c(t)$")
    ax1.plot(t[:-1] + dt / 2, ff_counts, "y:", linewidth=2, label="$N'_f(t)$")

    ax1.set_xlabel("Normalised Time")
    ax1.set_ylabel("Count")
    ax1.set_title("Cell Count Comparison")
    ax1.legend()

    # Plot volumes
    ax2.plot(t[:-1] + dt / 2, volumes_physical, "o", markersize=4, label="Raw volumes")
    ax2.plot(
        t2,
        s_cumulative_volumes.derivative()(t2) * dt,
        "r-",
        linewidth=2,
        label="$s'(V(t))$",
    )
    ax2.plot(t[:-1] + dt / 2, fb_volumes, "g--", linewidth=2, label="$V'_b(t)$")
    ax2.plot(t, fc_volumes, "m-.", linewidth=2, label="$V'_c(t)$")
    ax2.plot(t[:-1] + dt / 2, ff_volumes, "y:", linewidth=2, label="$V'_f(t)$")
    ax2.set_xlabel("Normalised Time")
    ax2.set_ylabel("Volume ($µm^3$)")
    ax2.set_title("Volume Comparison")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "derivative_methods_counts_volumes.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot density
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t[:-1] + dt / 2, raw_density, "o", markersize=4, label="Raw density")
    ax.axhline(avg_density, color="k", linestyle="--", label="Average density")
    ax.plot(t2, s_density_cumulative, "r-", linewidth=2, label="$s'(N(t))/s'(V(t))$")
    ax.plot(t[:-1] + dt / 2, fb_density, "g--", linewidth=2, label="$N'_b(t)/V'_b(t)$")
    ax.plot(t, fc_density, "m-.", linewidth=2, label="$N'_c(t)/V'_c(t)$")
    ax.plot(t[:-1] + dt / 2, ff_density, "y:", linewidth=2, label="$N'_f(t)/V'_f(t)$")
    ax.set_xlabel("Normalised Time")
    ax.set_ylabel("Density (cells/$µm^3$)")
    ax.set_title("Density Comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "derivative_methods_density.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
