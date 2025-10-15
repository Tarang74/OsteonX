"""
Node Density Analysis Example

This script analyzes the spatial distribution of osteocytes (nodes) within osteon structures.
It computes node density as a function of lamellar growth surfaces using smoothed spline interpolation.

The density is calculated as:
    ρ(t) = N'(t) / V'(t)
where N(t) is the cumulative node count and V(t) is the cumulative volume.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from osteonx import analysis, io, visuals, utils

plt.ioff()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyse osteocyte density in osteon structures"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing 3D image masks of osteon",
    )
    parser.add_argument(
        "--node-csv",
        type=str,
        default="node.csv",
        help="Node CSV filename (default: node.csv)",
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
    node_csv_path = Path(args.node_csv)

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

    # Import node data
    nodes = io.import_tina_nodes(
        csv_path=node_csv_path,
        osteon=osteon,
        filter_cells=True,
        filter_cellnodes=True,
    )

    # Compute EDTs
    dts = analysis.compute_edt(
        osteon,
        save_array=True,
        filename="osteon",
        path=arrays_output_path,
    )

    # Interpolate surfaces
    t, phi = analysis.interpolate_surfaces(osteon, dts, tsamples=args.tsamples)
    dt = t[1] - t[0]

    # Count nodes and measure volumes
    counts, volumes = analysis.find_density(phi, nodes)
    volumes_physical = utils.scale_to_physical(volumes, osteon)

    cumulative_counts = np.cumsum(np.hstack([0, counts]))
    cumulative_volumes_physical = np.cumsum(np.hstack([0, volumes_physical]))

    # Smooth counts and volumes separately
    t2, (s_cumulative_counts, s_cumulative_volumes) = utils.smooth(
        t,
        (cumulative_counts, cumulative_volumes_physical),
        lam=args.lam,
        upsample=args.upsample,
    )

    # Plot cumulative counts and volumes
    fig, (axs1, axs2) = plt.subplots(2, 2, figsize=(12, 10))

    # Cumulative counts
    axs1[0].plot(t, cumulative_counts, "o", markersize=4, label="$N(t)$")
    axs1[0].plot(t2, s_cumulative_counts(t2), "r-", linewidth=2, label="$s(N(t))$")
    axs1[0].set_xlabel("Normalised Time")
    axs1[0].set_ylabel("Count")
    axs1[0].set_title("Cumulative Node Count")
    axs1[0].legend()

    # Cumulative volumes
    axs1[1].plot(t, cumulative_volumes_physical, "o", markersize=4, label="$V(t)$")
    axs1[1].plot(t2, s_cumulative_volumes(t2), "r-", linewidth=2, label="$s(V(t))$")
    axs1[1].set_xlabel("Normalised Time")
    axs1[1].set_ylabel("Volume ($µm^3$)")
    axs1[1].set_title("Cumulative Volume")
    axs1[1].legend()

    # Counts
    axs2[0].plot(t[:-1], counts, "o", markersize=4, label="Raw counts")
    axs2[0].plot(
        t2,
        s_cumulative_counts.derivative()(t2) * dt,
        "r-",
        linewidth=2,
        label="$s'(N(t))$",
    )
    axs2[0].set_xlabel("Normalised Time")
    axs2[0].set_ylabel("Count")
    axs2[0].set_title("Node Count")
    axs2[0].legend()

    # Volumes
    axs2[1].plot(t[:-1], volumes_physical, "o", markersize=4, label="Raw volumes")
    axs2[1].plot(
        t2,
        s_cumulative_volumes.derivative()(t2) * dt,
        "r-",
        linewidth=2,
        label="$s'(V(t))$",
    )
    axs2[1].set_xlabel("Normalised Time")
    axs2[1].set_ylabel("Volume ($µm^3$)")
    axs2[1].set_title("Volume")
    axs2[1].legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "node_cumulative.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Density
    density_smooth = s_cumulative_counts.derivative()(
        t2
    ) / s_cumulative_volumes.derivative()(t2)

    # Plot density profile
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t[:-1], counts / volumes_physical, "o", markersize=4, label="Raw Density")
    ax.plot(t2, density_smooth, "r-", linewidth=2, label="$s'(N(t)) / s'(V(t))$")
    ax.set_xlabel("Normalised Time", fontsize=12)
    ax.set_ylabel("Density ($nodes/µm^3$)", fontsize=12)
    ax.set_title("Cell Density", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        str(figures_output_path / "node_density.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 3D visualization with nodes
    fig = visuals.plot_nodes_3d(
        nodes=nodes,
        phi=phi,
        out=str(figures_output_path / "nodes_3d.png"),
    )


if __name__ == "__main__":
    main()
