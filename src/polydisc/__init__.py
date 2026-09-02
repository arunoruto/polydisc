"""Polydisperse Poisson-disc sampling for discs, spheres and carpets.

A Numba-accelerated, variable-radius variant of Bridson's algorithm for
2D and 3D boxes, with an inter-particle spacing control and diagnostics
for what was actually realised.

Examples
--------
>>> import polydisc
>>> pack = polydisc.sample([100.0, 100.0], 4.0, seed=0)
>>> pack.n > 0
True

A flat carpet of polydisperse spheres, one average particle apart:

>>> spec = polydisc.RadiusSpec.lognormal(gmean=1.0, gstd=1.3)
>>> pack = polydisc.carpet(
...     (60.0, 60.0),
...     thickness=polydisc.monolayer_thickness(spec, spacing=2.0),
...     radii=spec,
...     spacing=2.0,
...     seed=0,
... )
>>> print(polydisc.report(pack))  # doctest: +ELLIPSIS
particles ...
"""

from __future__ import annotations

from polydisc.carpet import carpet, monolayer_thickness
from polydisc.diagnostics import (
    Report,
    gap_report,
    nearest_neighbour_gaps,
    projected_area_fraction,
    radius_report,
    report,
)
from polydisc.radii import RadiusSpec, as_radius_spec
from polydisc.sampling import (
    Packing,
    geometric_bias,
    poisson_disc_samples,
    sample,
)
from polydisc.spacing import apply_spacing, expected_gap, spacing_scale

__version__ = "0.2.0"

__all__ = [
    "Packing",
    "RadiusSpec",
    "Report",
    "__version__",
    "apply_spacing",
    "as_radius_spec",
    "carpet",
    "expected_gap",
    "gap_report",
    "geometric_bias",
    "monolayer_thickness",
    "nearest_neighbour_gaps",
    "poisson_disc_samples",
    "projected_area_fraction",
    "radius_report",
    "report",
    "sample",
    "spacing_scale",
]
