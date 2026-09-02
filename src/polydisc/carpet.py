"""Flat "carpets" of spheres: packings in a thin 3D slab.

A carpet is nothing more than a 3D packing in a box whose third edge is
comparable to a particle diameter, so this module is a convenience
layer over :func:`polydisc.sampling.sample` rather than a separate
algorithm.  Spheres pack in genuine 3D and may therefore overlap when
viewed from above; :func:`polydisc.diagnostics.projected_coverage`
quantifies how much.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from polydisc.radii import RadiusSpec, as_radius_spec
from polydisc.sampling import Packing, sample
from polydisc.spacing import spacing_scale


def carpet(
    extent: Sequence[float] | np.ndarray,
    thickness: float,
    radii: RadiusSpec | Sequence[float] | np.ndarray | float,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    repetitions: int = 30,
    seed: int | None = None,
    spacing: float = 0.0,
    fit_bounds: bool = True,
    max_points: int | None = None,
) -> Packing:
    """Generate a flat carpet of spheres in a thin slab.

    Parameters
    ----------
    extent : array_like
        In-plane edge lengths ``(Lx, Ly)`` of the carpet.
    thickness : float
        Slab depth along ``z``.  Must exceed the largest particle
        diameter; a value just above ``2 * rmax`` gives a monolayer,
        larger values a progressively thicker deposit.
    radii : RadiusSpec or array_like or float
        Radius distribution, as for :func:`polydisc.sampling.sample`.
    weights : array_like, optional
        Relative probabilities for a discrete `radii` sequence.
    repetitions : int, optional
        Candidates per active point.  Default 30.
    seed : int, optional
        Kernel seed; ``None`` draws one from NumPy's global generator.
    spacing : float, optional
        In-plane and through-plane clearance in units of the mean
        radius, applied as a position-only dilation to all three axes.
    fit_bounds : bool, optional
        ``True`` (default) keeps the finished carpet at exactly
        ``(Lx, Ly, thickness)``; ``False`` lets spacing grow the slab.
    max_points : int, optional
        Stop after this many particles.

    Returns
    -------
    Packing
        A 3D packing whose `bounds` are ``(Lx, Ly, thickness)``.

    Raises
    ------
    ValueError
        If `extent` does not have exactly two entries, `thickness` is
        not positive and finite, or the largest particle does not fit
        in the slab.

    See Also
    --------
    polydisc.sampling.sample : The general box sampler this wraps.
    polydisc.diagnostics.projected_coverage : Top-view area fraction.

    Examples
    --------
    >>> pack = carpet((80.0, 80.0), thickness=6.0, radii=2.0, seed=3)
    >>> pack.bounds.tolist()
    [80.0, 80.0, 6.0]
    >>> bool(np.all(pack.points[:, 2] >= 2.0))
    True
    """
    extent_arr = np.asarray(extent, dtype=np.float64).ravel()
    if extent_arr.shape[0] != 2:
        raise ValueError(
            f"extent must have exactly 2 entries (Lx, Ly), got {extent_arr.shape[0]}"
        )

    depth = float(thickness)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError(f"thickness must be a finite value > 0, got {thickness!r}")

    spec = as_radius_spec(radii, weights)
    box = np.array([extent_arr[0], extent_arr[1], depth], dtype=np.float64)

    return sample(
        box,
        spec,
        repetitions=repetitions,
        seed=seed,
        spacing=spacing,
        fit_bounds=fit_bounds,
        max_points=max_points,
    )


def monolayer_thickness(
    radii: RadiusSpec | Sequence[float] | np.ndarray | float,
    *,
    spacing: float = 0.0,
    fit_bounds: bool = True,
    clearance: float = 0.02,
) -> float:
    """Thinnest slab that still admits the largest particle.

    Parameters
    ----------
    radii : RadiusSpec or array_like or float
        The radius distribution the carpet will use.
    spacing : float, optional
        The `spacing` that will be passed to :func:`carpet`.  With
        `fit_bounds` true, spacing pre-shrinks the generation box on
        every axis, so the requested thickness must be pre-inflated by
        the same factor for the slab to stay a monolayer.  Default 0.
    fit_bounds : bool, optional
        The `fit_bounds` that will be passed to :func:`carpet`.  When
        false no pre-shrink happens and `spacing` is irrelevant here.
        Default True.
    clearance : float, optional
        Relative headroom above the bare minimum ``2 * rmax``.  A small
        positive value is required because a particle whose diameter
        exactly equals the slab depth has only one admissible centre
        height, which is unreachable in floating point.  Default 0.02.

    Returns
    -------
    float
        A `thickness` suitable for :func:`carpet`, giving the flattest
        carpet the distribution allows under those settings.

    Raises
    ------
    ValueError
        If `clearance` is not positive and finite, or `spacing` is
        negative.

    See Also
    --------
    carpet : Consumes the returned thickness.

    Examples
    --------
    >>> round(monolayer_thickness([1.0, 3.0]), 4)
    6.12
    >>> round(monolayer_thickness([1.0, 3.0], spacing=2.0), 4)
    12.24
    """
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise ValueError(f"clearance must be a finite value > 0, got {clearance!r}")
    spec = as_radius_spec(radii)
    inflate = spacing_scale(spacing) if fit_bounds else 1.0
    return 2.0 * spec.rmax * (1.0 + float(clearance)) * inflate
