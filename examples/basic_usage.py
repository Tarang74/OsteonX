from pathlib import Path
from osteonx import io, analysis, visuals

# 1. Import 3D image stack
osteon = io.import_masks(
    path=Path("masks"),
    downsample=(2, 2, 1),
    um_per_voxel=(0.379, 0.379, 0.296),
)

# 2. Compute Euclidean distance transforms
dts = analysis.compute_edt(osteon, path=Path("arrays"))

# 3. Interpolate surfaces
t, phi = analysis.interpolate_surfaces(osteon, dts, tsamples=20)

# 4. Visualise
visuals.plot_surfaces_2d(phi, layers=(phi.shape[2]//2,), out="figures/readme.png")
