"""
Segment Surface Density Analysis Example

This script analyses the spatial distribution of dendritic processes (segments) within osteon structures.
It computes segment surface density as a function of lamellar growth surfaces using smoothed spline interpolation.

The surface density is calculated as:
    ρ(t) = I(t) / A(t)
where I(t) is the segment intersection count and A(t) is the surface area.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from osteonx import analysis, io, visuals, utils

plt.ioff()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyse dendritic process surface density in osteon structures"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing 3D image masks of osteon",
    )
    parser.add_argument(
        "--segment-csv",
        type=str,
        default="segment.csv",
        help="Segment CSV filename (default: segment.csv)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsampling factor for images (default: 2)",
    )
    parser.add_argument(
        "--tsamples",
        type=int,
        default=20,
        help="Number of interpolated layers (default: 20)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.0005,
        dest="lam",
        help="Smoothing parameter (default: 0.0005)",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=5,
        help="Upsampling factor for smoothing (default: 5)",
    )
    parser.add_argument(
        "--array-dir",
        type=str,
        default="arrays",
        help="Output directory for arrays (default: arrays)",
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

    masks_path = Path(args.data_dir)
    segment_csv_path = Path(args.segment_csv)

    arrays_output_path = Path(args.array_dir)
    arrays_output_path.mkdir(parents=True, exist_ok=True)
    figures_output_path = Path(args.figure_dir)
    figures_output_path.mkdir(parents=True, exist_ok=True)

    # Import 3D mask stack
    osteon = io.import_masks(
        path=masks_path,
        pattern="*.png",
        downsample=(args.downsample, args.downsample, 1),
        um_per_voxel=(0.379, 0.379, 0.296),
        outer_mask_color=(0, 255, 0),
        inner_mask_color=(255, 0, 0),
    )

    segments, deltas = io.import_tina_segments(csv_path=segment_csv_path, osteon=osteon)

    # Compute EDTs
    dts = analysis.compute_edt(
        osteon,
        save_array=True,
        filename="osteon",
        path=arrays_output_path,
        parallel=0,
    )

    # Interpolate surfaces
    t, phi = analysis.interpolate_surfaces(osteon, dts, tsamples=args.tsamples)
    dt = t[1] - t[0]

    # Count segment intersections and measure surface areas
    counts, surface_areas = analysis.find_surface_density(phi, osteon, segments, deltas)
    surface_areas_physical = utils.scale_to_physical(surface_areas, osteon, dim=2)

    # Smooth counts and surface areas separately
    t2, (s_counts, s_surface_areas) = utils.smooth(
        t[1:-1],
        (counts, surface_areas_physical),
        lam=args.lam,
        upsample=args.upsample,
    )

    # Plot cumulative counts and surface areas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Counts
    ax1.plot(t[1:-1], counts, "o", markersize=4, label="I(t)")
    ax1.plot(
        t2,
        s_counts(t2),
        "r-",
        linewidth=2,
        label="$s(I(t))$",
    )
    ax1.set_xlabel("Normalised Time")
    ax1.set_ylabel("Count")
    ax1.set_title("Segment Count")
    ax1.legend()

    # Surface areas
    ax2.plot(
        t[1:-1], surface_areas_physical, "o", markersize=4, label="A(t)"
    )
    ax2.plot(
        t2,
        s_surface_areas(t2),
        "r-",
        linewidth=2,
        label="$s(A(t))$",
    )
    ax2.set_xlabel("Normalised Time")
    ax2.set_ylabel("Surface Area ($µm^2$)")
    ax2.set_title("Surface Area")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "segment_counts_areas.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Surface density
    surface_density_smooth = s_counts(t2) / s_surface_areas(t2)

    # Plot surface density profile
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        t[1:-1],
        counts / surface_areas_physical,
        "o",
        markersize=4,
        label="$\\rho_A(t)$",
    )
    ax.plot(t2, surface_density_smooth, "r-", linewidth=2, label="$\\tilde\\rho_A(t)$")
    ax.set_xlabel("Normalised Time", fontsize=12)
    ax.set_ylabel("Surface Density ($intersection/µm^2$)", fontsize=12)
    ax.set_title("Segment Intersection Surface Density", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "segment_surface_density.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 3D visualization with segments
    fig = visuals.plot_segments_3d(
        segments=segments,
        deltas=deltas,
        phi=phi,
        out=str(figures_output_path / "segment_3d.png"),
    )


if __name__ == "__main__":
    main()
