from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from edt import edt

from .types import Osteon


def _load(val):
    """Load a numpy array or return the array unchanged.

    Args:
        val: Either a path (str) to a .npy file or an ndarray.

    Returns:
        ndarray: Loaded array or the original input if already an array.
    """
    if isinstance(val, str):
        return np.load(val)
    return val


def compute_edt(
    osteon: Osteon,
    *,
    save_array: bool = True,
    filename: str = "osteon",
    path: Path = Path("arrays/"),
    parallel: int = 0,
) -> Dict:
    """Compute Euclidean distance transforms for an `Osteon`.

    Distances increase outward from the boundary.

    Args:
        osteon: Osteon dataclass with outer/inner boolean masks.
        save_array: If True save results as .npy files under `path`.
        filename: Base filename for saved arrays.
        path: Directory to save arrays.
        parallel: Parallel worker count for EDT computation.

    Returns:
        dict: Keys 'outer_dt','inner_dt','outer_dt_inv','inner_dt_inv','mask'.
              Values are file paths (if save_array) or numpy arrays.
    """
    outer_dt = edt(osteon.outer.astype(float), parallel=parallel)
    inner_dt = edt(osteon.inner.astype(float), parallel=parallel)
    outer_dt_inv = edt(np.logical_not(osteon.outer).astype(float), parallel=parallel)
    inner_dt_inv = edt(np.logical_not(osteon.inner).astype(float), parallel=parallel)
    mask = osteon.outer & osteon.inner

    if save_array:
        path.mkdir(parents=True, exist_ok=True)

        outer_path = str(path / f"{filename}-outer-dt.npy")
        inner_path = str(path / f"{filename}-inner-dt.npy")
        outer_inv_path = str(path / f"{filename}-outer-dt-inv.npy")
        inner_inv_path = str(path / f"{filename}-inner-dt-inv.npy")
        mask_path = str(path / f"{filename}-mask.npy")

        np.save(outer_path, outer_dt)
        np.save(inner_path, inner_dt)
        np.save(outer_inv_path, outer_dt_inv)
        np.save(inner_inv_path, inner_dt_inv)
        np.save(mask_path, mask)

        return {
            "outer_dt": outer_path,
            "inner_dt": inner_path,
            "outer_dt_inv": outer_inv_path,
            "inner_dt_inv": inner_inv_path,
            "mask": mask_path,
        }

    return {
        "outer_dt": outer_dt,
        "inner_dt": inner_dt,
        "outer_dt_inv": outer_dt_inv,
        "inner_dt_inv": inner_dt_inv,
        "mask": mask,
    }


def interpolate_surfaces(
    osteon: Osteon, dts: Dict, *, tsamples: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate level-set surfaces between outer and inner boundaries.

    Args:
        osteon: Osteon dataclass (used for shape).
        dts: Dict containing distance transforms (arrays or file paths).
        tsamples: Number of intermediate samples between boundaries.

    Returns:
        ndarray: t values of shape (tsamples,).
        ndarray: Phi array with shape (W, H, D, tsamples).
    """
    outer_dt = _load(dts["outer_dt"])
    inner_dt = _load(dts["inner_dt"])
    outer_dt_inv = _load(dts["outer_dt_inv"])
    inner_dt_inv = _load(dts["inner_dt_inv"])

    dt = 1 / tsamples
    tvals = np.linspace(0, 1, tsamples)

    phi = np.zeros([osteon.shape[0], osteon.shape[1], osteon.shape[2], tsamples])
    for ti, t in enumerate(tvals):
        if t == 0:
            phi[:, :, :, ti] = outer_dt - outer_dt_inv
        elif t == tvals[-1]:
            phi[:, :, :, ti] = -(inner_dt - inner_dt_inv)
        else:
            phi[:, :, :, ti] = (1 - t) * outer_dt - t * inner_dt

    return tvals, phi


def find_density(phi: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute node counts and volumes between layer defined by `phi`.

    Args:
        phi: (W, H, D, T) level-set array defining T layers.
        nodes: (N, 3) voxel coordinates (float or int).

    Returns:
        Tuple[ndarray, ndarray]: (counts, volumes) per layer.
    """
    tsamples = phi.shape[3] - 1
    counts = np.zeros(tsamples, dtype=int)
    volumes = np.zeros(tsamples, dtype=int)

    nodes_int = np.floor(nodes).astype(int)

    for ti in range(tsamples):
        region_mask = np.logical_and(
            phi[:, :, :, ti] >= 0, phi[:, :, :, ti + 1] < 0
        )
        counts[ti] = _count_nodes_in_region(region_mask, nodes_int)
        volumes[ti] = _find_volume(region_mask)

    return counts, volumes


def find_surface_density(
    phi: np.ndarray,
    osteon: Osteon,
    segments: np.ndarray,
    deltas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute segment intersection counts and surface areas.

    Args:
        phi: (W, H, D, T) level-set array defining T layers.
        osteon: Osteon.
        segments: (N, 3) segment start positions (voxels).
        deltas: (N, 3) segment deltas (voxels).

    Returns:
        Tuple[ndarray, ndarray]: (counts, areas) per layer.
    """
    tsamples = phi.shape[3] - 1
    counts = np.zeros(tsamples, dtype=int)
    areas = np.zeros(tsamples, dtype=float)

    segments_int = np.floor(segments).astype(int)

    for ti in range(tsamples):
        surface_mask = phi[:, :, :, ti]
        counts[ti] = _count_segment_intersections(surface_mask, segments_int, deltas)
        areas[ti] = _find_surface_area(surface_mask, osteon.um_per_voxel)

    return counts, areas


def _count_nodes_in_region(region_mask: np.ndarray, nodes: np.ndarray) -> int:
    """Count nodes that lie inside the boolean region mask.

    Args:
        region_mask: Boolean array of shape (W, H, D).
        nodes: (N, 3) integer voxel coordinates.

    Returns:
        int: Number of nodes inside the region.

    Raises:
        TypeError: if `nodes` is not of integer dtype.
    """
    if not np.issubdtype(nodes.dtype, np.integer):
        raise TypeError(
            f"nodes must have integer dtype, got {nodes.dtype}. "
            "Convert to integer coordinates before calling this function."
        )

    count = 0
    for x, y, z in nodes:
        xi, yi, zi = int(x), int(y), int(z)
        if (
            0 <= xi < region_mask.shape[0]
            and 0 <= yi < region_mask.shape[1]
            and 0 <= zi < region_mask.shape[2]
        ):
            if region_mask[xi, yi, zi]:
                count += 1
    return count


def _count_segment_intersections(
    surface_mask: np.ndarray,
    segments: np.ndarray,
    deltas: np.ndarray,
) -> int:
    """Count line segments intersections with a surface.

    Args:
        surface_mask: Boolean mask of shape (W, H, D).
        segments: (N, 3) segment start positions (voxels).
        deltas: (N, 3) segment deltas (voxels).

    Returns:
        int: Number of segment intersections.
    """
    count = 0
    for i in range(len(segments)):
        x1, y1, z1 = segments[i, :].astype(int)
        x2, y2, z2 = (segments[i, :] + deltas[i, :]).astype(int)

        if surface_mask[x1, y1, z1] * surface_mask[x2, y2, z2] <= 0:
            count += 1

    return count


def _find_volume(region_mask: np.ndarray) -> int:
    """Return voxel volume (count of True values) of a boolean mask.

    Args:
        region_mask: Boolean mask of shape (W, H, D).

    Returns:
        int: Number of True voxels.
    """
    return int(region_mask.sum())


def _find_surface_area(
    phi: np.ndarray, um_per_voxel: Tuple[float, float, float]
) -> float:
    """Estimate surface area from a level-set using marching cubes.

    Args:
        phi: 3D scalar field for marching cubes input.
        um_per_voxel: Micrometers per voxel.

    Returns:
        float: Surface area in physical units.
    """
    from skimage import measure

    verts, faces, _normals, _values = measure.marching_cubes(
        phi, level=0.0, spacing=um_per_voxel
    )
    return measure.mesh_surface_area(verts, faces)
