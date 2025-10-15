"""
Lamellar Growth Approximation Example

This script approximates lamellar growth surfaces in osteon structures using 3D image stack data.
It computes the Euclidean distance transform (EDT) of the Haversian canal and cement line,
and interpolates surfaces between them to approximate concentric lamellar layers.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from osteonx import analysis, io, visuals

plt.ioff()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Approximate lamellar growth surfaces in osteon structures"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing 3D image masks of osteon",
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
        default=10,
        help="Number of interpolated layers (default: 10)",
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

    # Setup paths
    masks_path = Path(args.data_dir)
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

    # Compute Euclidean distance transforms
    dts = analysis.compute_edt(
        osteon,
        save_array=True,
        filename="osteon",
        path=arrays_output_path,
        parallel=0,
    )

    # Interpolate surfaces between boundaries
    t, phi = analysis.interpolate_surfaces(osteon, dts, tsamples=args.tsamples)

    # Plot 2D cross-section
    z_slices = np.arange(0, osteon.shape[2], osteon.shape[2] // 5)
    visuals.plot_surfaces_2d(
        phi,
        layers=z_slices.tolist(),
        dims=(osteon.shape[0], osteon.shape[1]),
        out=str(figures_output_path / "lamellar_growth_2d.png"),
    )

    # Plot 3D visualization
    visuals.plot_surfaces_3d(
        phi=phi,
        out=str(figures_output_path / "lamellar_growth_3d.png"),
    )


if __name__ == "__main__":
    main()
