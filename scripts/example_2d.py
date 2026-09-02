"""Render a 2D polydisperse packing to ``out/example_2d.png``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

import polydisc

OUT = Path("out")


def main() -> None:
    """Sample a packing, draw it, and print its report."""
    spec = polydisc.RadiusSpec.lognormal(gmean=6.0, gstd=1.5)
    pack = polydisc.sample([600.0, 400.0], spec, repetitions=30, seed=0)
    print(polydisc.report(pack))

    fig, ax = plt.subplots(figsize=(12, 8))
    circles = [
        Circle((float(x), float(y)), float(r))
        for (x, y), r in zip(pack.points, pack.radii)
    ]
    collection = PatchCollection(circles, cmap="viridis", alpha=0.85)
    collection.set_array(pack.radii)
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax, label="radius")

    ax.set_xlim(0.0, float(pack.bounds[0]))
    ax.set_ylim(0.0, float(pack.bounds[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{pack.n} discs, lognormal radii (gmean 6, gstd 1.5), "
        f"fill {pack.fill_fraction:.3f}"
    )

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "example_2d.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'example_2d.png'}")


if __name__ == "__main__":
    main()
