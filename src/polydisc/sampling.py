"""Public sampling API: fill a region, get a :class:`Packing` back."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from polydisc import _kernels
from polydisc.radii import RadiusSpec, as_radius_spec
from polydisc.regions import Box, Region, as_region, region_from_metadata
from polydisc.spacing import spacing_scale

logger = logging.getLogger(__name__)

#: Absolute ceiling on the automatic `max_points` estimate, so a badly
#: conditioned request fails fast instead of exhausting memory.
_MAX_POINTS_CEILING = 200_000_000

#: Relative shift of the mean radius above which `sample` warns that the
#: container geometry, not the request, is choosing the sizes.
_BIAS_WARN_THRESHOLD = 0.05


@dataclass(frozen=True)
class Packing:
    """A generated set of non-overlapping discs or spheres.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres, ``D`` being 2 or 3.  Already scaled by
        `scale`.
    radii : (N,) numpy.ndarray
        Particle radii.  Never affected by `spacing`.
    region : Box or Ball
        The container the particles fill, after spacing.
    spec : RadiusSpec
        The radius distribution that was requested.
    spacing : float
        Requested clearance in mean radii; see
        :mod:`polydisc.spacing`.
    scale : float
        The position dilation actually applied, ``1 + spacing / 2``.
    fit_bounds : bool
        Whether generation was pre-shrunk so the finished packing
        occupies exactly the requested region.
    seed : int
        Seed handed to the kernel; replaying it reproduces `points`
        exactly.
    repetitions : int
        Candidates tried per active point (Bridson's ``k``).
    saturated : bool
        True when sampling ran until no active point remained, False
        when it stopped at ``max_points``.
    gap : float
        Requested minimum surface-to-surface distance between any two
        particles, in the units of the radii.

    Attributes
    ----------
    n : int
        Number of particles.
    dimensions : int
        2 or 3.
    bounds : numpy.ndarray
        Edge lengths of the region's axis-aligned bounding box.
    fill_fraction : float
        Total particle area/volume divided by the region's.

    Examples
    --------
    >>> from polydisc import sample
    >>> pack = sample([40.0, 40.0], 2.0, seed=1)
    >>> pack.dimensions
    2
    >>> bool(pack.n > 0)
    True
    """

    points: np.ndarray
    radii: np.ndarray
    region: Region
    spec: RadiusSpec = field(repr=False)
    spacing: float = 0.0
    scale: float = 1.0
    fit_bounds: bool = True
    seed: int = 0
    repetitions: int = 30
    saturated: bool = True
    gap: float = 0.0

    @property
    def n(self) -> int:
        """Number of particles in the packing."""
        return int(self.points.shape[0])

    @property
    def dimensions(self) -> int:
        """Spatial dimension of the packing (2 or 3)."""
        return int(self.points.shape[1])

    @property
    def bounds(self) -> np.ndarray:
        """Edge lengths of the region's axis-aligned bounding box."""
        return self.region.bounding_box

    @property
    def generation_region(self) -> Region:
        """Container the kernel actually sampled in, before dilation.

        Returns
        -------
        Box or Ball
            The region scaled down by `scale`.  Both `fit_bounds` modes
            end up here: fitting shrinks the request before sampling,
            expanding grows the container after it, and either way the
            region the kernel saw is the reported one undone by `scale`.
        """
        return self.region.scaled(1.0 / self.scale)

    @property
    def generation_bounds(self) -> np.ndarray:
        """Bounding box of :attr:`generation_region`."""
        return self.generation_region.bounding_box

    @property
    def fill_fraction(self) -> float:
        """Fraction of the region occupied by particles.

        Returns
        -------
        float
            Area fraction in 2D, volume fraction in 3D.  Particles are
            entirely inside the region -- or, across a periodic face,
            inside its tiling -- so this never exceeds one.
        """
        capacity = self.region.volume
        if capacity <= 0.0:
            return 0.0
        if self.dimensions == 2:
            occupied = float(np.sum(np.pi * self.radii**2))
        else:
            occupied = float(np.sum((4.0 / 3.0) * np.pi * self.radii**3))
        return occupied / capacity

    def metadata(self) -> dict[str, Any]:
        """Return the run's settings and summary as a plain mapping.

        Returns
        -------
        dict
            JSON-serialisable description of how the packing was made
            and what came out.
        """
        return {
            "n": self.n,
            "dimensions": self.dimensions,
            "region": self.region.describe(),
            "bounds": [float(b) for b in self.bounds],
            "spacing": float(self.spacing),
            "scale": float(self.scale),
            "fit_bounds": bool(self.fit_bounds),
            "seed": int(self.seed),
            "repetitions": int(self.repetitions),
            "saturated": bool(self.saturated),
            "gap": float(self.gap),
            "fill_fraction": self.fill_fraction,
            "radii": self.spec.describe(),
        }

    def save(self, path: str | Path) -> Path:
        """Write the packing to a ``.npz`` archive.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination.  Parent directories are created.

        Returns
        -------
        pathlib.Path
            The path written.

        See Also
        --------
        Packing.load : The inverse operation.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            points=self.points,
            radii=self.radii,
            bounds=self.bounds,
            metadata=np.asarray(json.dumps(self.metadata())),
        )
        return out

    @classmethod
    def load(cls, path: str | Path | IO[bytes]) -> Packing:
        """Read a packing back from a ``.npz`` archive.

        Parameters
        ----------
        path : str or pathlib.Path or file-like
            An archive written by :meth:`save`.  A binary file object is
            accepted too, so an upload can be read without touching
            disk.

        Returns
        -------
        Packing
            The restored packing.  `region` and `spec` are rebuilt from
            the stored metadata where possible, falling back to a box
            over the stored bounds and a discrete spec over the realised
            radii otherwise.

        Raises
        ------
        KeyError
            If the archive lacks the ``points``, ``radii`` or ``bounds``
            arrays.
        """
        source = Path(path) if isinstance(path, (str, Path)) else path
        with np.load(source, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float64)
            radii = np.asarray(data["radii"], dtype=np.float64)
            bounds = np.asarray(data["bounds"], dtype=np.float64)
            meta: dict[str, Any] = {}
            if "metadata" in data:
                meta = json.loads(str(data["metadata"]))

        region = region_from_metadata(meta.get("region")) or Box(bounds)
        return cls(
            points=points,
            radii=radii,
            region=region,
            spec=_spec_from_metadata(meta.get("radii"), radii),
            spacing=float(meta.get("spacing", 0.0)),
            scale=float(meta.get("scale", 1.0)),
            fit_bounds=bool(meta.get("fit_bounds", True)),
            seed=int(meta.get("seed", 0)),
            repetitions=int(meta.get("repetitions", 30)),
            saturated=bool(meta.get("saturated", True)),
            gap=float(meta.get("gap", 0.0)),
        )


def _spec_from_metadata(meta: dict[str, Any] | None, radii: np.ndarray) -> RadiusSpec:
    """Rebuild a :class:`RadiusSpec` from saved metadata."""
    if isinstance(meta, dict):
        kind = meta.get("kind")
        try:
            if kind == "discrete":
                return RadiusSpec.discrete(meta["values"], meta["weights"])
            if kind == "lognormal":
                return RadiusSpec.lognormal(
                    meta["gmean"],
                    meta["gstd"],
                    rmin=meta["rmin"],
                    rmax=meta["rmax"],
                )
            if kind == "normal":
                return RadiusSpec.normal(
                    meta["mean"],
                    meta["std"],
                    rmin=meta["rmin"],
                    rmax=meta["rmax"],
                )
            if kind == "uniform":
                return RadiusSpec.uniform(meta["rmin"], meta["rmax"])
        except (KeyError, ValueError):
            logger.warning(
                "Could not rebuild RadiusSpec from metadata; falling back "
                "to the realised radii."
            )
    unique = np.unique(radii) if radii.size else np.array([1.0])
    return RadiusSpec.discrete(unique)


def _auto_max_points(region: Region, rmin: float) -> int:
    """Hard geometric ceiling on how many particles can possibly fit.

    Particles lie entirely inside the region and do not overlap, so
    their areas/volumes sum to at most the region's.  That makes
    ``region / smallest_particle`` a true upper bound, which is what we
    use as the default ``max_points``: it never truncates a legitimate
    run, but it does stop a pathological request from growing without
    limit.
    """
    if region.dimensions == 2:
        unit = math.pi * rmin**2
    else:
        unit = (4.0 / 3.0) * math.pi * rmin**3
    return int(min(region.volume / unit + 8.0, _MAX_POINTS_CEILING))


def geometric_bias(
    spec: RadiusSpec, region: Region | Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Predict how a container's walls skew the realised size mix.

    A particle of radius ``r`` can only be centred in the sub-region its
    own size leaves free -- ``bounds - 2r`` inside a box, a ball of
    radius ``R - r`` inside a ball -- so large particles have
    proportionally fewer admissible positions than small ones.  In the
    dilute limit the realised density is the requested one reweighted by
    that admissible measure, which is exactly what this computes.  The
    effect is mild for a container much larger than the particles and
    severe for a thin slab, where it is the main reason a monolayer of a
    wide size distribution comes out almost entirely fine.

    Parameters
    ----------
    spec : RadiusSpec
        The requested radius distribution.
    region : Box or Ball or array_like
        The container that will be filled; a sequence is read as box
        edge lengths.

    Returns
    -------
    dict
        ``requested_mean``, ``admitted_mean``, ``bias`` (their relative
        difference) and ``excluded``, the probability mass whose radius
        cannot fit in the container at all.

    See Also
    --------
    polydisc.diagnostics.radius_report : What was realised in the end.

    Notes
    -----
    This is a boundary effect only.  Crowding adds a further bias in
    the same direction that no closed form predicts, so treat the
    result as a lower bound on the total distortion.

    Examples
    --------
    >>> spec = RadiusSpec.discrete([1.0, 3.0])
    >>> round(geometric_bias(spec, [100.0, 100.0])["bias"], 3)
    -0.021
    >>> round(geometric_bias(spec, [100.0, 100.0, 6.2])["bias"], 3)
    -0.458
    """
    container = as_region(region)
    values, pmf = spec.discretize()
    admissible = container.admissible_measure(values)

    requested_mean = float(np.sum(pmf * values))
    excluded = float(np.sum(pmf[admissible <= 0.0]))

    weighted = pmf * admissible
    total = float(np.sum(weighted))
    if total <= 0.0:
        return {
            "requested_mean": requested_mean,
            "admitted_mean": math.nan,
            "bias": math.nan,
            "excluded": 1.0,
        }

    admitted_mean = float(np.sum(weighted * values) / total)
    bias = (
        (admitted_mean - requested_mean) / requested_mean
        if requested_mean > 0
        else math.nan
    )
    return {
        "requested_mean": requested_mean,
        "admitted_mean": admitted_mean,
        "bias": bias,
        "excluded": excluded,
    }


def sample(
    region: Region | Sequence[float] | np.ndarray,
    radii: RadiusSpec | Sequence[float] | np.ndarray | float,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    repetitions: int = 30,
    seed: int | None = None,
    spacing: float = 0.0,
    fit_bounds: bool = True,
    max_points: int | None = None,
    gap: float = 0.0,
) -> Packing:
    """Fill a box or a ball with a polydisperse Poisson-disc packing.

    Parameters
    ----------
    region : Box or Ball or array_like
        The container to fill.  A plain sequence of edge lengths is read
        as a :class:`~polydisc.regions.Box`, two entries for 2D or three
        for 3D.  A flat carpet is a box whose third edge is close to a
        particle diameter.  ``Box(..., periodic=...)`` makes chosen axes
        wrap around, so the result tiles space along them.
    radii : RadiusSpec or array_like or float
        Radius distribution.  A float or sequence is read as a discrete
        set of allowed radii; use :class:`~polydisc.radii.RadiusSpec`
        for continuous (lognormal, normal, uniform) distributions.
    weights : array_like, optional
        Relative probabilities for a discrete `radii` sequence.
    repetitions : int, optional
        Bridson's ``k``: candidates tried around each active point
        before it is retired.  Higher values pack denser and cost more.
        Default 30.
    seed : int, optional
        Kernel seed.  ``None`` draws one from NumPy's global generator,
        so seeding that makes the result reproducible too.  The seed
        actually used is recorded on the returned packing.
    spacing : float, optional
        Clearance between neighbouring surfaces in units of the mean
        radius, applied as a position-only dilation.  ``2`` leaves room
        for one average particle.  See :mod:`polydisc.spacing`.
    fit_bounds : bool, optional
        With `spacing` above zero, ``True`` (default) generates in a
        pre-shrunk container so the finished packing occupies exactly
        `region`; ``False`` generates in `region` and lets the container
        grow by `scale`, matching PyFracVAL's behaviour.
    max_points : int, optional
        Stop after this many particles.  Defaults to a geometric upper
        bound that never truncates a real run.
    gap : float, optional
        Minimum surface-to-surface distance between any two particles,
        in the units of the radii: every pair ends up with
        ``|c_i - c_j| >= r_i + r_j + gap``, across periodic faces too
        (minimum image).  Unlike `spacing` this is a hard constraint
        during sampling, not a dilation afterwards; the packing still
        saturates.  Walls are not affected.  Default 0.

    Returns
    -------
    Packing
        The generated particles plus the settings that produced them.

    Raises
    ------
    ValueError
        If `region` is malformed, `repetitions` or `max_points` is
        below one, `spacing` or `gap` is negative, the largest possible
        particle cannot fit in the (possibly pre-shrunk) container, or a
        periodic edge is shorter than ``2 * (2 * rmax + gap)`` (the
        minimum-image limit).

    See Also
    --------
    polydisc.diagnostics.report : Check what was actually realised.
    geometric_bias : Predict the container's effect before sampling.

    Notes
    -----
    Candidate radii are drawn once per *attempt*, so a dense region
    accepts small particles more readily than large ones and the
    realised size distribution is biased slightly small relative to the
    request.  This is inherent to the dart-throwing formulation, not a
    defect of this implementation; the bias is reported by
    :func:`polydisc.diagnostics.radius_report` rather than hidden.

    Examples
    --------
    >>> pack = sample([100.0, 100.0], [2.0, 4.0], seed=7)
    >>> pack.points.shape[1]
    2
    >>> float(pack.radii.min()) >= 2.0
    True

    Filling a ball instead of a box:

    >>> from polydisc.regions import Ball
    >>> pack = sample(Ball(30.0), 3.0, seed=7)
    >>> bool(pack.region.contains(pack.points, pack.radii).all())
    True

    A box periodic in x with a minimum gap of 0.5 between surfaces:

    >>> from polydisc.regions import Box
    >>> pack = sample(Box([60.0, 30.0], periodic=(True, False)), 1.0,
    ...               gap=0.5, seed=7)
    >>> from polydisc.diagnostics import nearest_neighbour_gaps
    >>> bool(nearest_neighbour_gaps(pack.points, pack.radii, pack.region).min()
    ...      >= 0.5 - 1e-9)
    True
    """
    container = as_region(region)
    spec = as_radius_spec(radii, weights)

    if int(repetitions) < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")

    gap = float(gap)
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError(f"gap must be a finite value >= 0, got {gap!r}")

    scale = spacing_scale(spacing)
    gen_region = container.scaled(1.0 / scale) if fit_bounds else container
    out_region = container if fit_bounds else container.scaled(scale)
    # The dilation by `scale` widens gaps by at least `scale`, so sampling
    # with gap / scale still leaves `gap` in the finished packing.
    gen_gap = gap / scale
    periodic = getattr(gen_region, "periodic", (False,) * gen_region.dimensions)

    limit = gen_region.max_particle_radius
    if spec.rmax >= limit:
        hint = ""
        if fit_bounds and scale > 1.0:
            hint = (
                f" (spacing={spacing} shrinks the container by {scale:g}x "
                "before sampling; pass fit_bounds=False to keep the full "
                "size)"
            )
        raise ValueError(
            f"largest particle (radius {spec.rmax:g}) does not fit in the "
            f"container, which admits at most {limit:g}{hint}"
        )

    for axis, length in enumerate(gen_region.bounding_box):
        if periodic[axis] and length < 2.0 * (2.0 * spec.rmax + gen_gap):
            raise ValueError(
                f"periodic edge {'xyz'[axis]} ({length:g} while sampling) is "
                f"shorter than 2 * (2 * rmax + gap) = "
                f"{2.0 * (2.0 * spec.rmax + gen_gap):g}; neighbours across "
                "the face would be ambiguous"
            )

    if max_points is None:
        max_points_int = _auto_max_points(gen_region, spec.rmin)
    else:
        max_points_int = int(max_points)
        if max_points_int < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points}")

    bias = geometric_bias(spec, gen_region)
    if abs(bias["bias"]) > _BIAS_WARN_THRESHOLD:
        logger.warning(
            "Container geometry skews the size distribution: it leaves "
            "large particles few admissible positions, shifting the mean "
            "radius %.1f%% (requested %.4g, admissible %.4g). Enlarge the "
            "container, narrow the radius range, or expect a small-biased "
            "packing.",
            bias["bias"] * 100.0,
            bias["requested_mean"],
            bias["admitted_mean"],
        )

    if seed is None:
        seed_int = int(np.random.randint(0, 2**31 - 1))
    else:
        seed_int = int(seed) % (2**31 - 1)

    box = gen_region.bounding_box
    args = (
        gen_region.kind,
        spec.kind,
        spec.values,
        spec.cdf,
        spec.p0,
        spec.p1,
        spec.rmin,
        spec.rmax,
        int(repetitions),
        seed_int,
        max_points_int,
        gen_gap,
        *periodic,
    )

    if gen_region.dimensions == 2:
        points, out_radii, status = _kernels._sample_2d(box[0], box[1], *args)
    else:
        points, out_radii, status = _kernels._sample_3d(box[0], box[1], box[2], *args)

    if status == _kernels.STATUS_MAX_POINTS:
        logger.warning(
            "Stopped at max_points=%d before the packing saturated.",
            max_points_int,
        )

    if scale != 1.0:
        points = points * scale

    return Packing(
        points=points,
        radii=out_radii,
        region=out_region,
        spec=spec,
        spacing=float(spacing),
        scale=scale,
        fit_bounds=bool(fit_bounds),
        seed=seed_int,
        repetitions=int(repetitions),
        saturated=status == _kernels.STATUS_SATURATED,
        gap=gap,
    )


def poisson_disc_samples(
    region: Region | Sequence[float] | np.ndarray,
    radii: RadiusSpec | Sequence[float] | np.ndarray | float,
    repetitions: int = 30,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    seed: int | None = None,
    spacing: float = 0.0,
    fit_bounds: bool = True,
    gap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a packing and return just the arrays.

    A thin wrapper over :func:`sample` for callers that only want the
    geometry.

    Parameters
    ----------
    region : Box or Ball or array_like
        The container to fill; a sequence is read as box edge lengths.
    radii : RadiusSpec or array_like or float
        Radius distribution, as for :func:`sample`.
    repetitions : int, optional
        Candidates per active point.  Default 30.
    weights : array_like, optional
        Relative probabilities for a discrete `radii` sequence.
    seed : int, optional
        Kernel seed; ``None`` draws one from NumPy's global generator.
    spacing : float, optional
        Clearance in mean radii.
    fit_bounds : bool, optional
        See :func:`sample`.
    gap : float, optional
        Minimum surface gap; see :func:`sample`.

    Returns
    -------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.

    See Also
    --------
    sample : Returns a :class:`Packing` with the run's metadata.

    Examples
    --------
    >>> pts, rr = poisson_disc_samples([50.0, 50.0], [3.0], seed=0)
    >>> pts.shape[0] == rr.shape[0]
    True
    """
    pack = sample(
        region,
        radii,
        weights=weights,
        repetitions=repetitions,
        seed=seed,
        spacing=spacing,
        fit_bounds=fit_bounds,
        gap=gap,
    )
    return pack.points, pack.radii
