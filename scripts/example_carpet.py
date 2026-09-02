"""Compare carpet spacings and render them to ``out/example_carpet.png``.

A carpet needs no special support: it is a :class:`polydisc.Box` whose
depth is about one particle diameter.  The one wrinkle is that spacing
pre-shrinks every axis, so the depth must be inflated by the same factor
for the slab to stay a monolayer -- that is the ``spacing_scale`` term
below.

The three panels show the same carpet at three spacings, in top view and
side view, so the position-only dilation is visible: the particles keep
their sizes while the layer opens up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import polydisc

OUT = Path("out")
SPACINGS = (0.0, 1.0, 2.0)
EXTENT = (120.0, 120.0)

#: Width of the x-slice shown in the side views.  A side view of the
#: whole 120-wide carpet at true aspect would be a few pixels tall, and
#: stretching it instead turns every sphere into a misleading ellipse.
SLICE_WIDTH = 30.0


def _draw(ax, xs, ys, radii, xlim, ylim, vmin, vmax) -> None:
    """Draw one projection of a carpet at true aspect ratio."""
    circles = [Circle((float(x), float(y)), float(r)) for x, y, r in zip(xs, ys, radii)]
    collection = PatchCollection(circles, cmap="viridis", alpha=0.85)
    collection.set_array(radii)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    """Sample one carpet per spacing and plot top and side views."""
    spec = polydisc.RadiusSpec.lognormal(gmean=1.0, gstd=1.3)

    packs = []
    for spacing in SPACINGS:
        # 2.05 * rmax is the thinnest slab that still admits the largest
        # particle; the spacing factor undoes the pre-shrink.
        depth = 2.05 * spec.rmax * polydisc.spacing_scale(spacing)
        pack = polydisc.sample([*EXTENT, depth], spec, spacing=spacing, seed=0)
        report = polydisc.report(pack)
        print(f"--- spacing {spacing} ---")
        print(report)
        print()
        packs.append((spacing, pack, report))

    # One colour scale across all panels, so a colour means the same
    # radius everywhere and the panels are actually comparable.
    vmin = min(float(p.radii.min()) for _, p, _ in packs)
    vmax = max(float(p.radii.max()) for _, p, _ in packs)
    max_depth = max(float(p.bounds[2]) for _, p, _ in packs)

    fig, axes = plt.subplots(
        2,
        len(SPACINGS),
        figsize=(5 * len(SPACINGS), 7.5),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for column, (spacing, pack, report) in enumerate(packs):
        _draw(
            axes[0, column],
            pack.points[:, 0],
            pack.points[:, 1],
            pack.radii,
            (0.0, float(pack.bounds[0])),
            (0.0, float(pack.bounds[1])),
            vmin,
            vmax,
        )
        axes[0, column].set_title(
            f"spacing {spacing:g}  -  {pack.n} spheres\n"
            f"projected area {report.projected_area_fraction:.3f}"
        )

        keep = pack.points[:, 0] <= SLICE_WIDTH
        _draw(
            axes[1, column],
            pack.points[keep, 0],
            pack.points[keep, 2],
            pack.radii[keep],
            (0.0, SLICE_WIDTH),
            (0.0, max_depth),
            vmin,
            vmax,
        )
        axes[1, column].set_xlabel(
            f"side view, x < {SLICE_WIDTH:g}, true aspect "
            f"(slab {pack.bounds[2]:.2f} deep)"
        )

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "example_carpet.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT / 'example_carpet.png'}")


if __name__ == "__main__":
    main()
