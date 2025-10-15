from dataclasses import dataclass
from typing import Tuple

from numpy import ndarray


@dataclass(frozen=True)
class Osteon:
    outer: ndarray
    inner: ndarray
    shape: Tuple[int, int, int]
    um_per_voxel: Tuple[float, float, float]
    downsample: Tuple[int, int, int]
