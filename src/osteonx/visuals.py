from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def plot_surfaces_2d(
    phi: np.ndarray,
    layers: Tuple[int] = tuple(),
    nodes: np.ndarray | None = None,
    dims=None,
    out: str | None = None,
):
    """Plot zero-level contours for a level-set array.

    Args:
        phi: 4D array with shape (X, Y, Z, T).
        layers: Z slice indices to plot.
        nodes: Optional (N, 3) nodes to overlay.
        dims: Optional (width, height) axis limits.
        out: Optional file path to save the figure.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if not layers:
        layers = (phi.shape[2] // 2,)

    n_layers = len(layers)
    n_cols = int(np.ceil(np.sqrt(n_layers)))
    n_rows = int(np.ceil(n_layers / n_cols))
    tsamples = phi.shape[3]

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

    for i, layer in enumerate(layers):
        if n_layers == 1:
            ax = axs
        else:
            ax = axs[i // n_cols][i % n_cols]

        if nodes is not None:
            x, y, z = nodes.T
            ax.plot(x, y, ".", markersize=0.5)

        for ti in range(tsamples):
            ax.contour(phi[:, :, layer, ti], levels=[0], alpha=0.5)

        ax.set_title(f"Layer {layer}")
        ax.set_aspect(1)
        if dims:
            ax.set_xlim(0, dims[0])
            ax.set_ylim(0, dims[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")

    return fig, axs


def plot_surfaces_3d(
    phi: np.ndarray,
    distance: float = 800,
    out: str | None = None,
    opacity: float = 0.3,
    colormap: str = "jet",
    azimuth: float = 135,
    elevation: float = 60,
    perspective: bool = True,
):
    """Render zero-level surfaces from a 4D level-set array in 3D using mayavi.

    Args:
        phi: 4D level-set array with shape (X, Y, Z, T).
        distance: Camera distance for the view.
        out: Optional file path to save an offscreen render.
        opacity: Opacity for the rendered surfaces (0..1).
        colormap: Mayavi colormap name to use for surfaces.

    Returns:
        mayavi.mlab.Figure or None: The mayavi figure if rendering succeeded.
    """
    f = mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))

    tcount = phi.shape[3]
    for ti in range(tcount):
        mlab.contour3d(
            phi[:, :, :, ti],
            contours=[0],
            opacity=opacity,
            colormap=colormap,
        )

    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
    f.scene.parallel_projection = not perspective  # type: ignore

    if out:
        mlab.savefig(out, size=(1000, 800))
        mlab.close(all=True)
    else:
        mlab.show()

    return f


def plot_nodes_3d(
    nodes: np.ndarray,
    phi: np.ndarray | None = None,
    distance: float = 800,
    out: str | None = None,
    opacity: float = 0.3,
    colormap: str = "jet",
    azimuth: float = 0,
    elevation: float = 180,
    roll: float = -90,
    perspective: bool = True,
):
    """Render nodes and optional zero-level surfaces from a 4D level-set array in 3D using mayavi.

    Args:
        nodes: (N, 3) node coordinates.
        phi: Optional 4D level-set array for contour surfaces.
        distance: Camera distance for the view.
        out: Optional file path to save an offscreen render.
        opacity: Opacity for the rendered surfaces (0..1).
        colormap: Mayavi colormap name to use for surfaces.

    Returns:
        mayavi.mlab.Figure or None: The mayavi figure if rendering succeeded.
    """
    f = mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))

    # Plot nodes as points
    x, y, z = nodes.T
    mlab.points3d(x, y, z, scale_factor=1, color=(0, 0, 0), opacity=0.5)

    # Plot surfaces
    if phi is not None:
        tcount = phi.shape[3]
        for ti in range(tcount):
            mlab.contour3d(
                phi[:, :, :, ti],
                contours=[0],
                opacity=opacity,
                colormap=colormap,
            )

    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
    mlab.roll(roll)
    f.scene.parallel_projection = not perspective  # type: ignore

    if out:
        mlab.savefig(out, size=(1000, 800))
        mlab.close(all=True)
    else:
        mlab.show()

    return f


def plot_segments_3d(
    segments: np.ndarray,
    deltas: np.ndarray,
    phi: np.ndarray | None = None,
    distance: float = 800,
    out: str | None = None,
    opacity: float = 0.3,
    colormap: str = "jet",
    azimuth: float = 0,
    elevation: float = 180,
    roll: float = -90,
    perspective: bool = True,
):
    """Render segments and optional zero-level surfaces from a 4D level-set array in 3D using mayavi.

    Args:
        segments: (N, 3) segment start coordinates.
        deltas: (N, 3) segment direction vectors.
        phi: Optional 4D level-set array for contour surfaces.
        distance: Camera distance for the view.
        out: Optional file path to save an offscreen render.
        opacity: Opacity for the rendered surfaces (0..1).
        colormap: Mayavi colormap name to use for surfaces.

    Returns:
        mayavi.mlab.Figure or None: The mayavi figure if rendering succeeded.
    """
    f = mlab.figure(size=(1000, 800), bgcolor=(1, 1, 1))

    # Plot nodes as points
    mlab.quiver3d(
        segments[:, 0],
        segments[:, 1],
        segments[:, 2],
        deltas[:, 0],
        deltas[:, 1],
        deltas[:, 2],
        scale_factor=1,
        color=(0.2, 0.2, 0.8),
        opacity=0.3,
    )

    # Plot surfaces if phi is provided
    if phi is not None:
        tcount = phi.shape[3]
        for ti in range(tcount):
            mlab.contour3d(
                phi[:, :, :, ti],
                contours=[0],
                opacity=opacity,
                colormap=colormap,
            )

    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)
    mlab.roll(roll)
    f.scene.parallel_projection = not perspective  # type: ignore

    if out:
        mlab.savefig(out, size=(1000, 800))
        mlab.close(all=True)
    else:
        mlab.show()

    return f


def plot_nodes_with_surface_2d(
    nodes: np.ndarray,
    phi: np.ndarray,
    layer: int,
    dims=None,
    out: str | None = None,
):
    """Plot nodes overlaid on zero-level contours for a level-set array in 2D.

    Args:
        phi: 4D array with shape (X, Y, Z, T).
        nodes: (N, 3) node coordinates.
        layer: Z slice index to plot.
        dims: Optional (width, height) axis limits.
        out: Optional file path to save the figure.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plot_surfaces_2d(phi, layers=(layer,), dims=dims)

    # Filter nodes to the specified layer
    nodes_in_layer = nodes[np.abs(nodes[:, 2] - layer) < 5]
    ax.scatter(nodes_in_layer[:, 1], nodes_in_layer[:, 0], c="red", s=10)

    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


def plot_segments_with_surface_2d(
    segments: np.ndarray,
    deltas: np.ndarray,
    phi: np.ndarray,
    layer: int,
    dims=None,
    out: str | None = None,
):
    """Plot segments overlaid on zero-level contours for a level-set array in 2D.

    Args:
        phi: 4D array with shape (X, Y, Z, T).
        segments: (N, 3) segment start coordinates.
        deltas: (N, 3) segment direction vectors.
        layer: Z slice index to plot.
        dims: Optional (width, height) axis limits.
        out: Optional file path to save the figure.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plot_surfaces_2d(phi, layers=(layer,), dims=dims)

    # Filter segments to the specified layer
    segments_in_layer = segments[np.abs(segments[:, 2] - layer) < 5]
    deltas_in_layer = deltas[np.abs(segments[:, 2] - layer) < 5]

    # Quiver plot for segments
    ax.quiver(
        segments_in_layer[:, 1],
        segments_in_layer[:, 0],
        deltas_in_layer[:, 1],
        deltas_in_layer[:, 0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.5,
    )

    # Plot segment start points
    ax.scatter(segments_in_layer[:, 1], segments_in_layer[:, 0], c="red", s=5)

    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig, ax
