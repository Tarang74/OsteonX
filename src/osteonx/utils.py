import numpy as np
from typing import Tuple
from .types import Osteon
from scipy.interpolate import BSpline, make_smoothing_spline


def scale_to_physical(
    measures: np.ndarray,
    osteon: Osteon,
    dim: int | None = None,
) -> np.ndarray:
    """Convert voxel counts/areas/volumes to physical units.

    Args:
        measures: Array of measures in voxel units.
        osteon: Osteon.

    Returns:
        ndarray: Measures scaled to physical units.

    Raises:
        ValueError: if scalar spacing is provided without `dim`.
    """
    max_dim = dim if dim is not None else len(osteon.um_per_voxel)
    scale_factor = np.prod(osteon.um_per_voxel[:max_dim]) * np.prod(
        osteon.downsample[:max_dim]
    )
    return measures * scale_factor


def difference(
    data: np.ndarray, method: str, prepend: float | None = None, append: float | None = None
) -> np.ndarray:
    """Compute the first- or second-order discrete difference of a 1D array.

    Args:
        data: Input 1D array.
        method: Difference method ('forward', 'backward', 'central').
        prepend: Value to prepend for 'backward' difference. Usually 0.
        append: Value to append for 'forward' difference. Usually the last value.

    Returns:
        ndarray: Array of differences.
    """
    if method == "forward":
        if append is not None:
            return np.diff(data, append=append)[:-1]
        else:
            raise ValueError("Append value must be provided for forward difference.")
    elif method == "backward":
        if prepend is not None:
            return np.diff(data, prepend=prepend)[1:]
        else:
            raise ValueError("Prepend value must be provided for backward difference.")
    elif method == "central":
        return np.gradient(data)
    else:
        raise ValueError(
            "Invalid method. Choose from 'forward', 'backward', 'central'."
        )


def smooth(
    t: np.ndarray,
    data: Tuple[np.ndarray, ...],
    lam: float = 0.0005,
    upsample: int = 5,
) -> Tuple[np.ndarray, Tuple[BSpline, ...]]:
    """Apply smoothing spline to two data points.

    Args:
        t: Time or normalised parameter array.
        data: Data arrays to be smoothed.
        lam: Smoothing parameter (higher -> smoother).
        upsample: Output upsampling factor.

    Returns:
        Tuple[ndarray, Tuple[ndarray]]: (t_out, (smoothed_data...)) where t_out is the upsampled parameter array.
    """
    t_out = np.linspace(0, 1, len(t) * upsample)
    smoothed_data = tuple(make_smoothing_spline(t, d, lam=lam) for d in data)

    return (t_out, smoothed_data)
