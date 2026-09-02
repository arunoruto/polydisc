"""Figures for the polydisc dashboard.

Two plotting libraries, chosen per job rather than by preference:
Matplotlib draws circles at true scale, which is the only honest way to
show a packing and the only one that stays fast past a few thousand
particles; Plotly handles the interactive 3D view and the histograms,
where hovering and zooming are the point.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from polydisc.diagnostics import nearest_neighbour_gaps
from polydisc.regions import Ball
from polydisc.sampling import Packing


def _lifted_viridis(floor: float = 0.18, steps: int = 256):
    """Viridis with its near-black low end trimmed off.

    The smallest particles land at the bottom of the scale, and viridis
    ends in a purple dark enough to vanish against a dark background.
    Trimming the colormap rather than shifting the colour limits keeps
    the colourbar labelled with real radii.

    Parameters
    ----------
    floor : float, optional
        Fraction of the scale to drop from the dark end.  Default 0.18.
    steps : int, optional
        Number of samples in the rebuilt colormap.  Default 256.

    Returns
    -------
    colormap : matplotlib.colors.Colormap
        For the Matplotlib figures.
    colorscale : list
        The same ramp as Plotly ``[position, "rgb(...)"]`` pairs.
    """
    samples = np.linspace(floor, 1.0, steps)
    colours = plt.get_cmap("viridis")(samples)
    colormap = LinearSegmentedColormap.from_list("viridis_lifted", colours)

    positions = np.linspace(0.0, 1.0, 16)
    colorscale = [
        [
            float(position),
            "rgb({}, {}, {})".format(
                *(round(255 * channel) for channel in colormap(position)[:3])
            ),
        ]
        for position in positions
    ]
    return colormap, colorscale


#: Sequential colour scale shared by every figure, so a colour means the
#: same radius wherever it appears, in either plotting library.
CMAP, COLORSCALE = _lifted_viridis()

#: Transparent backgrounds and mid-grey text let one figure sit on
#: either Streamlit theme; the app does not force light or dark, so the
#: figures must not assume one.
_PAPER = "rgba(0,0,0,0)"
_GRID = "rgba(128,128,128,0.25)"
_INK = "0.5"

#: Figure width in inches, and the range its height may take.  A
#: cross-section of a carpet is thirty times wider than it is deep;
#: sizing the figure at that true aspect leaves a sliver a few pixels
#: tall with axis labels larger than the plot itself.  Clamping the
#: figure while leaving the axes at equal aspect letterboxes the data
#: instead, which stays readable and still shows circles as circles.
_FIG_WIDTH = 9.0
_FIG_HEIGHT_LIMITS = (2.5, 11.0)


def _figure(width: float, height: float):
    """Create a figure whose height is clamped to a readable range."""
    inches = _FIG_WIDTH * float(height) / float(width)
    return plt.subplots(
        figsize=(_FIG_WIDTH, float(np.clip(inches, *_FIG_HEIGHT_LIMITS)))
    )


#: Approximate on-screen extent, in pixels, of the longest axis of the
#: 3D scene.  Only used to turn radii in data units into Plotly's
#: pixel-valued marker sizes; being roughly right is enough.
_SCENE_PIXELS = 450.0


def _transparent(figure, *axes) -> None:
    """Make a Matplotlib figure theme-agnostic."""
    figure.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.xaxis.label.set_color(_INK)
        ax.yaxis.label.set_color(_INK)
        ax.tick_params(colors=_INK)


def subsample(
    pack: Packing, limit: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Take at most `limit` particles, uniformly at random.

    Parameters
    ----------
    pack : Packing
        The packing to draw from.
    limit : int
        Maximum number of particles to keep.
    seed : int, optional
        Seed for the choice, so a redraw is stable.  Default 0.

    Returns
    -------
    points : numpy.ndarray
        The kept centres.
    radii : numpy.ndarray
        The kept radii.
    reduced : bool
        True when particles were dropped.
    """
    if pack.n <= limit:
        return pack.points, pack.radii, False
    index = np.random.default_rng(seed).choice(pack.n, size=limit, replace=False)
    index.sort()
    return pack.points[index], pack.radii[index], True


def _outline(ax, pack: Packing) -> None:
    """Draw the container boundary on a 2D axis."""
    if isinstance(pack.region, Ball) and pack.dimensions == 2:
        ax.add_patch(
            Circle(
                tuple(pack.region.centre),
                pack.region.radius,
                fill=False,
                edgecolor="0.45",
                linewidth=1.2,
                zorder=3,
            )
        )
    else:
        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                float(pack.bounds[0]),
                float(pack.bounds[1]),
                fill=False,
                edgecolor="0.45",
                linewidth=1.2,
                zorder=3,
            )
        )


def _circles(ax, points, radii, vmin, vmax, alpha=0.9):
    """Add particles as a single PatchCollection and return it.

    One collection, not one artist per particle: an artist each is what
    makes a 100k-particle figure take minutes instead of seconds.
    """
    circles = [
        Circle((float(x), float(y)), float(r))
        for (x, y), r in zip(points[:, :2], radii)
    ]
    collection = PatchCollection(circles, cmap=CMAP, alpha=alpha, linewidths=0)
    collection.set_array(radii)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    return collection


def plot_2d(pack: Packing, limit: int = 40_000) -> tuple[Figure, bool]:
    """Draw a 2D packing at true scale.

    Parameters
    ----------
    pack : Packing
        A two-dimensional packing.
    limit : int, optional
        Maximum particles drawn before subsampling.  Default 40000.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The rendered figure.
    reduced : bool
        True when the view shows a random subset.
    """
    points, radii, reduced = subsample(pack, limit)

    figure, ax = _figure(float(pack.bounds[0]), float(pack.bounds[1]))
    collection = _circles(
        ax, points, radii, float(pack.radii.min()), float(pack.radii.max())
    )
    _outline(ax, pack)

    margin = 0.02 * float(pack.bounds[0])
    ax.set_xlim(-margin, float(pack.bounds[0]) + margin)
    ax.set_ylim(-margin, float(pack.bounds[1]) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(collection, ax=ax, label="radius", fraction=0.035, pad=0.02)
    bar.ax.yaxis.label.set_color(_INK)
    bar.ax.tick_params(colors=_INK)
    _transparent(figure, ax)
    figure.tight_layout()
    return figure, reduced


def plot_slice(
    pack: Packing, axis: int, centre: float, thickness: float, limit: int = 40_000
) -> tuple[Figure, int]:
    """Draw a slab of a 3D packing as true-scale circles.

    Every particle intersecting the slab is drawn at the radius of its
    intersection with the slab's mid-plane, so the picture is a real
    cross-section rather than a projection.

    Parameters
    ----------
    pack : Packing
        A three-dimensional packing.
    axis : int
        Axis normal to the slab: 0, 1 or 2.
    centre : float
        Position of the slab's mid-plane along `axis`.
    thickness : float
        Full depth of the slab.
    limit : int, optional
        Maximum particles drawn before subsampling.  Default 40000.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The rendered cross-section.
    shown : int
        How many particles intersect the slab.
    """
    others = [i for i in range(3) if i != axis]
    offset = np.abs(pack.points[:, axis] - centre)
    keep = offset < pack.radii + 0.5 * thickness

    points = pack.points[keep][:, others]
    # Radius of the circle where the sphere meets the mid-plane.
    cut = np.sqrt(np.maximum(pack.radii[keep] ** 2 - offset[keep] ** 2, 0.0))
    shown = int(points.shape[0])

    if shown > limit:
        index = np.random.default_rng(0).choice(shown, size=limit, replace=False)
        index.sort()
        points, cut = points[index], cut[index]

    width = float(pack.bounds[others[0]])
    height = float(pack.bounds[others[1]])
    figure, ax = _figure(width, height)

    if shown:
        collection = _circles(
            ax, points, cut, float(pack.radii.min()), float(pack.radii.max())
        )
        bar = figure.colorbar(
            collection, ax=ax, label="radius", fraction=0.035, pad=0.02
        )
        bar.ax.yaxis.label.set_color(_INK)
        bar.ax.tick_params(colors=_INK)

    ax.add_patch(
        Rectangle(
            (0.0, 0.0), width, height, fill=False, edgecolor="0.45", linewidth=1.2
        )
    )
    margin = 0.02 * width
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(-margin, height + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("xyz"[others[0]])
    ax.set_ylabel("xyz"[others[1]])
    ax.set_xticks([])
    ax.set_yticks([])
    _transparent(figure, ax)
    figure.tight_layout()
    return figure, shown


def plot_3d(pack: Packing, limit: int = 20_000) -> tuple[go.Figure, bool]:
    """Draw a 3D packing as an interactive point cloud.

    Parameters
    ----------
    pack : Packing
        A three-dimensional packing.
    limit : int, optional
        Maximum particles drawn before subsampling.  Default 20000.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        The rendered figure.
    reduced : bool
        True when the view shows a random subset.

    Notes
    -----
    Marker sizes are in screen pixels, so they track relative radii but
    do not rescale as you zoom; use :func:`plot_slice` for a view at
    true geometric scale.
    """
    points, radii, reduced = subsample(pack, limit)
    span = float(np.max(pack.bounds))
    # Plotly marker size is a diameter in screen pixels, so convert from
    # data units: the longest axis occupies roughly _SCENE_PIXELS of the
    # rendered scene. Clamped so a handful of huge particles cannot blot
    # out the view and a very wide container still leaves them visible.
    sizes = np.clip(2.0 * radii / span * _SCENE_PIXELS, 2.0, 40.0)

    figure = go.Figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={
                "size": sizes,
                "color": radii,
                "colorscale": COLORSCALE,
                # Shortened and centred so it clears the modebar.
                "colorbar": {
                    "title": "radius",
                    "thickness": 12,
                    "len": 0.7,
                    "y": 0.45,
                },
                "opacity": 0.9,
                "line": {"width": 0},
            },
            hovertemplate=(
                "x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}"
                "<br>r %{marker.color:.4f}<extra></extra>"
            ),
        )
    )

    axis = {
        "showbackground": False,
        "gridcolor": _GRID,
        "zeroline": False,
        "showticklabels": False,
        "title": "",
    }
    figure.update_layout(
        scene={
            "xaxis": {**axis, "range": [0, pack.bounds[0]]},
            "yaxis": {**axis, "range": [0, pack.bounds[1]]},
            "zaxis": {**axis, "range": [0, pack.bounds[2]]},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=640,
        paper_bgcolor=_PAPER,
    )
    return figure, reduced


def plot_radius_histogram(pack: Packing, show_requested: bool) -> go.Figure:
    """Compare realised radii against the requested distribution.

    Parameters
    ----------
    pack : Packing
        The packing to summarise.
    show_requested : bool
        Whether to overlay the requested density.  Off for an uploaded
        file, whose spec was inferred from the data and would compare
        against itself.

    Returns
    -------
    plotly.graph_objects.Figure
        A density histogram with an optional reference curve.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=pack.radii,
            histnorm="probability density",
            nbinsx=min(60, max(10, int(np.sqrt(pack.n)))),
            name="realised",
            marker={"color": "#4c78a8"},
        )
    )

    if show_requested:
        values, pmf = pack.spec.discretize(160)
        width = float(values[1] - values[0]) if values.size > 1 else 1.0
        figure.add_trace(
            go.Scatter(
                x=values,
                y=pmf / width,
                mode="lines",
                name="requested",
                line={"color": "#f58518", "width": 2.5},
            )
        )

    figure.update_layout(
        xaxis_title="radius",
        yaxis_title="density",
        bargap=0.02,
        height=340,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(gridcolor=_GRID)
    figure.update_yaxes(gridcolor=_GRID)
    return figure


def plot_gap_histogram(pack: Packing) -> go.Figure:
    """Show nearest-neighbour clearances in units of the mean radius.

    Parameters
    ----------
    pack : Packing
        The packing to summarise.

    Returns
    -------
    plotly.graph_objects.Figure
        A histogram with the requested spacing marked, so it is
        immediately visible whether the dilation delivered it.
    """
    gaps = nearest_neighbour_gaps(pack.points, pack.radii)
    finite = gaps[np.isfinite(gaps)] / float(np.mean(pack.radii))

    figure = go.Figure(
        go.Histogram(
            x=finite,
            nbinsx=min(80, max(10, int(np.sqrt(max(finite.size, 1))))),
            marker={"color": "#54a24b"},
            name="gap",
        )
    )
    if pack.spacing > 0:
        figure.add_vline(
            x=pack.spacing,
            line={"color": "#f58518", "width": 2, "dash": "dash"},
            annotation_text=f"requested spacing {pack.spacing:g}",
            annotation_position="top right",
        )
    figure.update_layout(
        xaxis_title="nearest-neighbour gap / mean radius",
        yaxis_title="particles",
        bargap=0.02,
        height=340,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        showlegend=False,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
    )
    figure.update_xaxes(gridcolor=_GRID)
    figure.update_yaxes(gridcolor=_GRID)
    return figure
