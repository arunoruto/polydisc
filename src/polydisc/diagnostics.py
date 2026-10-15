"""Check what a packing actually realised, rather than what was asked.

Two things routinely differ from the request and are easy to miss:

Size distribution
    Candidate radii are drawn once per placement *attempt*, so in a
    crowded neighbourhood a small candidate is likelier to survive than
    a large one.  The realised distribution is therefore biased towards
    small radii, more strongly at high fill fractions and for wide
    distributions.  :func:`radius_report` measures that bias.

Spacing
    :mod:`polydisc.spacing` dilates positions by ``1 + spacing / 2``,
    which opens ``spacing * r_mean`` between two touching *average*
    particles.  Real neighbours are neither touching nor average, so
    :func:`gap_report` reports the gaps that were actually obtained.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polydisc._kernels import _build_grid, _nearest_gaps
from polydisc.radii import RadiusSpec
from polydisc.regions import Region, as_region
from polydisc.sampling import Packing, geometric_bias

#: Gaps more negative than this (relative to the mean radius) count as
#: genuine overlaps rather than floating-point noise.
OVERLAP_TOLERANCE = 1e-9


def nearest_neighbour_gaps(
    points: np.ndarray, radii: np.ndarray, region: Region | None = None
) -> np.ndarray:
    """Surface-to-surface distance from each particle to its closest.

    Parameters
    ----------
    points : (N, D) array_like
        Particle centres, ``D`` being 2 or 3.
    radii : (N,) array_like
        Particle radii.
    region : Box or Ball, optional
        The container.  Only a :class:`~polydisc.regions.Box` with
        periodic axes changes the result: neighbours across its periodic
        faces count too (minimum image).

    Returns
    -------
    numpy.ndarray
        ``(N,)`` gaps ``|x_i - x_j| - r_i - r_j`` minimised over
        ``j != i``.  Negative entries mean overlap.  A packing of fewer
        than two particles yields all-``inf``.

    Raises
    ------
    ValueError
        If the shapes disagree or `points` is not 2D or 3D.

    Notes
    -----
    Exact, not approximate: the search walks outward in cell shells and
    stops only once no unvisited shell could contain a closer surface.
    Periodic axes are handled by adding the shifted copies of every
    particle (``3 ** p`` copies for ``p`` periodic axes) and keeping the
    originals' results.

    Examples
    --------
    >>> pts = np.array([[0.0, 0.0], [5.0, 0.0]])
    >>> nearest_neighbour_gaps(pts, np.array([1.0, 2.0]))
    array([2., 2.])
    """
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64).ravel()

    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if rad.shape[0] != pts.shape[0]:
        raise ValueError(
            f"radii length {rad.shape[0]} does not match {pts.shape[0]} points"
        )

    n = pts.shape[0]
    if n < 2:
        return np.full(n, np.inf, dtype=np.float64)

    periodic = getattr(region, "periodic", ())
    if any(periodic):
        bounds = region.bounding_box
        steps = [(-1, 0, 1) if p else (0,) for p in periodic]
        shifts = [np.asarray(s) * bounds for s in itertools.product(*steps) if any(s)]
        pts = np.concatenate([pts, *(pts + s for s in shifts)])
        rad = np.tile(rad, len(shifts) + 1)
        return nearest_neighbour_gaps(pts, rad)[:n]

    if pts.shape[1] == 2:
        pts3 = np.zeros((n, 3), dtype=np.float64)
        pts3[:, :2] = pts
    else:
        pts3 = pts.astype(np.float64, copy=True)

    pts3 -= pts3.min(axis=0)
    pts3 = np.ascontiguousarray(pts3)

    rmax = float(np.max(rad))
    cell = 2.0 * rmax
    extent = pts3.max(axis=0)
    nx = max(1, math.ceil(extent[0] / cell))
    ny = max(1, math.ceil(extent[1] / cell))
    nz = max(1, math.ceil(extent[2] / cell))

    head, nxt = _build_grid(pts3, cell, nx, ny, nz)
    return _nearest_gaps(pts3, rad, cell, nx, ny, nz, head, nxt, rmax)


def gap_report(
    points: np.ndarray, radii: np.ndarray, region: Region | None = None
) -> dict[str, Any]:
    """Summarise nearest-neighbour clearances in a packing.

    Parameters
    ----------
    points : (N, D) array_like
        Particle centres.
    radii : (N,) array_like
        Particle radii.
    region : Box or Ball, optional
        The container; periodic box axes are measured with the minimum
        image, see :func:`nearest_neighbour_gaps`.

    Returns
    -------
    dict
        ``min``, ``mean``, ``median``, ``p05`` and ``p95`` gaps; the
        same five divided by the mean radius under ``*_over_r``;
        ``overlaps``, the number of particles whose closest neighbour
        intersects them; and ``worst_overlap``, the deepest such
        intersection (``0`` when there are none).

    See Also
    --------
    nearest_neighbour_gaps : The per-particle values summarised here.
    """
    rad = np.asarray(radii, dtype=np.float64).ravel()
    gaps = nearest_neighbour_gaps(points, rad, region)
    finite = gaps[np.isfinite(gaps)]

    if finite.size == 0:
        return {
            "min": math.inf,
            "mean": math.inf,
            "median": math.inf,
            "p05": math.inf,
            "p95": math.inf,
            "min_over_r": math.inf,
            "mean_over_r": math.inf,
            "median_over_r": math.inf,
            "p05_over_r": math.inf,
            "p95_over_r": math.inf,
            "overlaps": 0,
            "worst_overlap": 0.0,
        }

    r_mean = float(np.mean(rad)) if rad.size else 1.0
    tol = -OVERLAP_TOLERANCE * max(r_mean, 1.0)
    overlapping = finite[finite < tol]

    stats = {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }
    for key in ("min", "mean", "median", "p05", "p95"):
        stats[f"{key}_over_r"] = stats[key] / r_mean if r_mean > 0 else math.inf

    stats["overlaps"] = int(overlapping.size)
    stats["worst_overlap"] = float(-np.min(overlapping)) if overlapping.size else 0.0
    return stats


def _spec_moments(spec: RadiusSpec, samples: int = 4096) -> tuple[float, float]:
    """Arithmetic mean and standard deviation of a spec."""
    values, pmf = spec.discretize(samples)
    mean = float(np.sum(pmf * values))
    var = float(np.sum(pmf * (values - mean) ** 2))
    return mean, math.sqrt(max(var, 0.0))


def radius_report(radii: np.ndarray, spec: RadiusSpec | None = None) -> dict[str, Any]:
    """Compare the realised radii against the requested distribution.

    Parameters
    ----------
    radii : (N,) array_like
        Realised particle radii.
    spec : RadiusSpec, optional
        The distribution that was requested.  When given, the result
        gains ``requested_mean``, ``requested_std``, ``mean_bias``
        (realised minus requested, in units of the requested mean) and
        ``ks``, the largest gap between the empirical and requested
        CDFs.

    Returns
    -------
    dict
        Realised ``n``, ``mean``, ``std``, ``min``, ``max``, ``gmean``
        and ``gstd``, plus the comparison keys described above.

    Notes
    -----
    A negative ``mean_bias`` of a few percent is expected and is the
    dart-throwing bias described in this module's docstring, not an
    error.  Large values (``|mean_bias| > 0.1``) usually mean the box
    is too crowded or too thin for the largest particles requested.
    """
    rad = np.asarray(radii, dtype=np.float64).ravel()
    if rad.size == 0:
        return {"n": 0}

    log_r = np.log(rad)
    out: dict[str, Any] = {
        "n": int(rad.size),
        "mean": float(np.mean(rad)),
        "std": float(np.std(rad)),
        "min": float(np.min(rad)),
        "max": float(np.max(rad)),
        "gmean": float(np.exp(np.mean(log_r))),
        "gstd": float(np.exp(np.std(log_r))),
    }

    if spec is None:
        return out

    req_mean, req_std = _spec_moments(spec)
    out["requested_mean"] = req_mean
    out["requested_std"] = req_std
    out["mean_bias"] = (out["mean"] - req_mean) / req_mean if req_mean > 0 else math.nan

    out["ks"] = _ks_statistic(rad, spec)
    return out


def _ks_statistic(radii: np.ndarray, spec: RadiusSpec) -> float:
    """Largest gap between the empirical and requested CDFs.

    Compares both sides of every distinct value.  A discrete spec puts
    atoms in the target CDF, and comparing only the right-continuous
    value there reports a large distance for a perfect match -- a
    monodisperse packing would score 1.0.  Taking the left limit as
    well (via the next representable float below each value) makes the
    statistic zero for an exact match in both the discrete and the
    continuous case.
    """
    values, counts = np.unique(radii, return_counts=True)
    cumulative = np.cumsum(counts)

    empirical_at = cumulative / radii.size
    empirical_before = (cumulative - counts) / radii.size
    target_at = spec.cdf_at(values)
    target_before = spec.cdf_at(np.nextafter(values, -np.inf))

    return float(
        max(
            np.max(np.abs(empirical_at - target_at)),
            np.max(np.abs(empirical_before - target_before)),
        )
    )


def projected_area_fraction(
    radii: np.ndarray, region: Region | Sequence[float] | np.ndarray
) -> float:
    """Summed disc area of a packing divided by the region's footprint.

    Parameters
    ----------
    radii : (N,) array_like
        Particle radii.
    region : Box or Ball or array_like
        The container; a sequence is read as box edge lengths, so a
        packing's ``bounds`` can be passed directly.

    Returns
    -------
    float
        ``sum(pi * r**2)`` divided by the area of the region's
        projection onto the first two axes.

    Notes
    -----
    This counts overlapping projections twice and can exceed one for a
    container deeper than a monolayer.  That is deliberate: it is the
    quantity that enters single-scattering estimates, whereas the true
    covered fraction saturates at one and hides the difference.
    """
    rad = np.asarray(radii, dtype=np.float64).ravel()
    footprint = as_region(region).footprint
    if footprint <= 0.0:
        return 0.0
    return float(np.sum(np.pi * rad**2) / footprint)


@dataclass(frozen=True)
class Report:
    """Diagnostics for one packing.

    Parameters
    ----------
    n : int
        Particle count.
    dimensions : int
        2 or 3.
    region : dict
        The container, from ``Box.describe`` or ``Ball.describe``.
    region_label : str
        One-line description of the container.
    bounds : list of float
        Edge lengths of the container's bounding box.
    fill_fraction : float
        Area/volume fraction occupied by particles.
    projected_area_fraction : float
        Top-view summed disc area over footprint; 3D packings only.
    radii : dict
        Output of :func:`radius_report`.
    boundary_bias : dict
        Output of :func:`polydisc.sampling.geometric_bias` for the
        container that was sampled: the part of the size bias
        attributable to the walls alone.
    gaps : dict
        Output of :func:`gap_report`.
    spacing : float
        Requested spacing in mean radii.
    saturated : bool
        Whether sampling ran to completion.
    """

    n: int
    dimensions: int
    region: dict[str, Any]
    region_label: str
    bounds: list[float]
    fill_fraction: float
    projected_area_fraction: float
    radii: dict[str, Any]
    boundary_bias: dict[str, Any]
    gaps: dict[str, Any]
    spacing: float
    saturated: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping.

        Returns
        -------
        dict
            Every field, with NumPy scalars already converted.
        """
        return {
            "n": self.n,
            "dimensions": self.dimensions,
            "region": self.region,
            "region_label": self.region_label,
            "bounds": self.bounds,
            "fill_fraction": self.fill_fraction,
            "projected_area_fraction": self.projected_area_fraction,
            "radii": self.radii,
            "boundary_bias": self.boundary_bias,
            "gaps": self.gaps,
            "spacing": self.spacing,
            "saturated": self.saturated,
        }

    def __str__(self) -> str:
        lines = [
            f"particles         {self.n}",
            f"dimensions        {self.dimensions}D",
            f"region            {self.region_label}",
            f"fill fraction     {self.fill_fraction:.4f}",
        ]
        if self.dimensions == 3:
            lines.append(f"projected area    {self.projected_area_fraction:.4f}")
        if not self.saturated:
            lines.append("saturated         no (stopped at max_points)")

        r = self.radii
        if r.get("n"):
            lines.append(
                f"radius            mean {r['mean']:.4g}  "
                f"std {r['std']:.4g}  "
                f"range [{r['min']:.4g}, {r['max']:.4g}]"
            )
            lines.append(
                f"radius (geom)     gmean {r['gmean']:.4g}  gstd {r['gstd']:.4g}"
            )
            if "requested_mean" in r:
                lines.append(
                    f"vs requested      mean {r['requested_mean']:.4g}  "
                    f"bias {r['mean_bias'] * 100:+.2f}%  "
                    f"KS {r['ks']:.3f}"
                )
            wall = self.boundary_bias.get("bias")
            if wall is not None and abs(wall) > 0.01:
                lines.append(
                    f"  of which walls  {wall * 100:+.2f}% "
                    "(container geometry, not crowding)"
                )

        g = self.gaps
        if math.isfinite(g.get("min", math.inf)):
            lines.append(
                f"neighbour gap     min {g['min']:.4g}  "
                f"median {g['median']:.4g}  "
                f"mean {g['mean']:.4g}"
            )
            lines.append(
                f"gap / mean radius min {g['min_over_r']:.3f}  "
                f"median {g['median_over_r']:.3f}  "
                f"(spacing requested {self.spacing:g})"
            )
            if g["overlaps"]:
                lines.append(
                    f"OVERLAPS          {g['overlaps']} particles, "
                    f"worst {g['worst_overlap']:.3g}"
                )
        return "\n".join(lines)


def report(packing: Packing) -> Report:
    """Build the full diagnostic report for a packing.

    Parameters
    ----------
    packing : Packing
        The packing to inspect.

    Returns
    -------
    Report
        Realised size distribution, neighbour gaps and fill fractions.
        Printing the result gives a readable summary.

    Examples
    --------
    >>> from polydisc import sample
    >>> pack = sample([60.0, 60.0], 3.0, seed=0)
    >>> rep = report(pack)
    >>> rep.gaps["overlaps"]
    0
    """
    projected = 0.0
    if packing.dimensions == 3:
        projected = projected_area_fraction(packing.radii, packing.region)

    return Report(
        n=packing.n,
        dimensions=packing.dimensions,
        region=packing.region.describe(),
        region_label=packing.region.label,
        bounds=[float(b) for b in packing.bounds],
        fill_fraction=packing.fill_fraction,
        projected_area_fraction=projected,
        radii=radius_report(packing.radii, packing.spec),
        boundary_bias=geometric_bias(packing.spec, packing.generation_region),
        gaps=gap_report(packing.points, packing.radii, packing.region),
        spacing=float(packing.spacing),
        saturated=bool(packing.saturated),
    )
