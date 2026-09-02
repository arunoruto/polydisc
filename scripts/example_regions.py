"""Render a box and a ball packing to ``out/example_regions.png``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import polydisc

OUT = Path("out")


def _draw(ax, pack: polydisc.Packing, vmin: float, vmax: float) -> None:
    """Draw a 2D packing and outline its container."""
    circles = [
        Circle((float(x), float(y)), float(r))
        for (x, y), r in zip(pack.points, pack.radii)
    ]
    collection = PatchCollection(circles, cmap="viridis", alpha=0.85)
    collection.set_array(pack.radii)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection)

    if isinstance(pack.region, polydisc.Ball):
        ax.add_patch(
            Circle(
                tuple(pack.region.centre),
                pack.region.radius,
                fill=False,
                edgecolor="0.3",
                linewidth=1.2,
            )
        )
    else:
        ax.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                float(pack.bounds[0]),
                float(pack.bounds[1]),
                fill=False,
                edgecolor="0.3",
                linewidth=1.2,
            )
        )

    ax.set_xlim(-2.0, float(pack.bounds[0]) + 2.0)
    ax.set_ylim(-2.0, float(pack.bounds[1]) + 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    """Fill a box and a ball with the same size distribution."""
    spec = polydisc.RadiusSpec.lognormal(gmean=6.0, gstd=1.5)
    packs = [
        polydisc.sample(polydisc.Box([400.0, 400.0]), spec, seed=0),
        polydisc.sample(polydisc.Ball(200.0, dimensions=2), spec, seed=0),
    ]

    vmin = min(float(p.radii.min()) for p in packs)
    vmax = max(float(p.radii.max()) for p in packs)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, pack in zip(axes, packs):
        report = polydisc.report(pack)
        print(report)
        print()
        _draw(ax, pack, vmin, vmax)
        ax.set_title(
            f"{pack.region.label}\n{pack.n} discs, fill {pack.fill_fraction:.3f}"
        )

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "example_regions.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT / 'example_regions.png'}")


if __name__ == "__main__":
    main()
