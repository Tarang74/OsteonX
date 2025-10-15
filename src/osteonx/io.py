from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .types import Osteon


def import_masks(
    path: Path,
    *,
    pattern: str = "*.png",
    downsample: Tuple[int, int, int] = (1, 1, 1),
    um_per_voxel: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    outer_mask_color: Tuple[int, int, int] = (0, 255, 0),
    inner_mask_color: Tuple[int, int, int] = (255, 0, 0),
) -> Osteon:
    """Load a stack of PNG masks and build 3D boolean masks.

    Args:
        path: Directory containing mask images.
        pattern: Glob pattern to match files.
        downsample: Integer downsample factor applied to images.
        um_per_voxel: Micrometers per voxel.
        outer_mask_color: RGB color tuple for outer mask pixels.
        inner_mask_color: RGB color tuple for inner mask pixels.

    Returns:
        Osteon: dataclass containing outer/inner masks and metadata.

    Raises:
        FileNotFoundError: if no matching image files are found.
    """
    p = Path(path)
    image_files = sorted(p.glob(pattern))
    if not image_files:
        raise FileNotFoundError(f"No files matching {pattern} in {path}")

    first_image = np.array(Image.open(image_files[0]).convert("RGB"))
    first_image = np.flipud(first_image)
    first_image = first_image[::downsample[0], ::downsample[1]]

    shape = (first_image.shape[0], first_image.shape[1], len(image_files))

    outer = np.zeros(shape, dtype=bool)
    inner = np.zeros(shape, dtype=bool)

    for z, image_file in enumerate(image_files):
        image = np.array(Image.open(image_file).convert("RGB"))
        image = np.flipud(image)

        image = image[::downsample[0], ::downsample[1]]

        outer[:, :, z] = np.logical_or(
            np.all(image == outer_mask_color, axis=-1),
            np.all(image == inner_mask_color, axis=-1),
        )
        inner[:, :, z] = ~np.all(image == outer_mask_color, axis=-1)

    return Osteon(
        outer=outer,
        inner=inner,
        shape=shape,
        um_per_voxel=um_per_voxel,
        downsample=downsample,
    )


def import_tina_nodes(
    csv_path: Path,
    osteon: Osteon,
    filter_cells: bool = True,
    filter_cellnodes: bool = True,
):
    """Import node data exported from TINA (CSV) and convert to voxel
    coordinates.

    Args:
        csv_path: Path to the node CSV file.
        osteon: Osteon.
        filter_cells: Keep only rows where `cell`==1 when True.
        filter_cellnodes: Keep only rows where `cellnode`==1 when True.

    Returns:
        ndarray: N x 3 array of node coordinates in voxel units.
    """
    df = pd.read_csv(csv_path)

    if filter_cells and filter_cellnodes:
        df = df[np.logical_or(df["cell"] == 1, df["cellnode"] == 1)]
    elif filter_cells:
        df = df[df["cell"] == 1]
    elif filter_cellnodes:
        df = df[df["cellnode"] == 1]

    Dx = osteon.um_per_voxel[0] * osteon.downsample[0]
    Dy = osteon.um_per_voxel[1] * osteon.downsample[1]
    Dz = osteon.um_per_voxel[2] * osteon.downsample[2]

    X = df["x"] / Dx
    X = osteon.shape[0] - X
    Y = df["y"] / Dy
    Z = df["z"] / Dz

    nodes = np.vstack((X, Y, Z)).T
    return nodes


def import_tina_segments(
    csv_path: Path,
    osteon: Osteon,
):
    """Import segment/edge data exported from TINA (CSV) and convert to voxels.

    Args:
        csv_path: Path to the segment CSV file.
        osteon: Osteon.

    Returns:
        Tuple[ndarray, ndarray]: (segments, deltas) in voxel coordinates.
    """
    df = pd.read_csv(csv_path)

    Dx = osteon.um_per_voxel[0] * osteon.downsample[0]
    Dy = osteon.um_per_voxel[1] * osteon.downsample[1]
    Dz = osteon.um_per_voxel[2] * osteon.downsample[2]

    x = df["x"] / Dx
    x = osteon.shape[0] - x
    y = df["y"] / Dy
    z = df["z"] / Dz
    segments = np.vstack((x, y, z)).T

    dx = -df["dx"] * df["seglength"] / Dx
    dy = df["dy"] * df["seglength"] / Dy
    dz = df["dz"] * df["seglength"] / Dz
    deltas = np.vstack((dx, dy, dz)).T

    return segments, deltas
