"""Polydisperse Poisson-disc sampling for discs and spheres.

A Numba-accelerated, variable-radius variant of Bridson's algorithm.
Particles fill a region -- a :class:`~polydisc.regions.Box` or a
:class:`~polydisc.regions.Ball` -- in 2D or 3D, with an inter-particle
spacing control and diagnostics for what was actually realised.

Examples
--------
>>> import polydisc
>>> pack = polydisc.sample([100.0, 100.0], 4.0, seed=0)
>>> pack.n > 0
True

A flat carpet is a box one particle deep, with the spacing set so an
average particle fits between neighbours:

>>> spec = polydisc.RadiusSpec.lognormal(gmean=1.0, gstd=1.3)
>>> spacing = 2.0
>>> depth = 2.05 * spec.rmax * polydisc.spacing_scale(spacing)
>>> pack = polydisc.sample(
...     [60.0, 60.0, depth], spec, spacing=spacing, seed=0
... )
>>> print(polydisc.report(pack))  # doctest: +ELLIPSIS
particles ...

A ball packed with spheres:

>>> pack = polydisc.sample(polydisc.Ball(30.0), 2.0, seed=0)
>>> bool(pack.region.contains(pack.points, pack.radii).all())
True
"""

from __future__ import annotations

from polydisc.diagnostics import (
    Report,
    gap_report,
    nearest_neighbour_gaps,
    projected_area_fraction,
    radius_report,
    report,
)
from polydisc.radii import RadiusSpec, as_radius_spec
from polydisc.regions import Ball, Box, Region, as_region
from polydisc.sampling import (
    Packing,
    geometric_bias,
    poisson_disc_samples,
    sample,
)
from polydisc.spacing import apply_spacing, expected_gap, spacing_scale

__version__ = "0.4.0"

__all__ = [
    "Ball",
    "Box",
    "Packing",
    "RadiusSpec",
    "Region",
    "Report",
    "__version__",
    "apply_spacing",
    "as_radius_spec",
    "as_region",
    "expected_gap",
    "gap_report",
    "geometric_bias",
    "nearest_neighbour_gaps",
    "poisson_disc_samples",
    "projected_area_fraction",
    "radius_report",
    "report",
    "sample",
    "spacing_scale",
]
