"""Inter-particle spacing as a position-only scale factor.

Spacing is expressed the way it is in PyFracVAL and YASF: *how many
mean radii of clearance should sit between neighbouring particles*.
A spacing of ``2`` therefore leaves room for one average particle
(``2 * r_mean``, one average diameter) between two average neighbours.

The transformation is a pure dilation of the positions — radii are
never touched::

    T = 1 + spacing / 2
    positions *= T

For two touching particles ``i`` and ``j`` this opens a gap of
``(T - 1) * (r_i + r_j) = spacing * (r_i + r_j) / 2``, which is
``spacing * r_mean`` for an average pair.  Because the map is a single
global dilation it preserves the structure of the packing exactly: every
pairwise distance is scaled by the same factor, so the pair correlation
function is merely stretched, not reshaped.
"""

from __future__ import annotations

import math

import numpy as np


def spacing_scale(spacing: float) -> float:
    """Convert a spacing in mean radii into a position scale factor.

    Parameters
    ----------
    spacing : float
        Requested clearance between neighbouring particle surfaces, in
        units of the mean radius.  Must be finite and ``>= 0``.

    Returns
    -------
    float
        The factor ``T = 1 + spacing / 2``, always ``>= 1``.

    Raises
    ------
    ValueError
        If `spacing` is negative or not finite.

    Examples
    --------
    >>> spacing_scale(0.0)
    1.0
    >>> spacing_scale(2.0)
    2.0
    """
    value = float(spacing)
    if not math.isfinite(value):
        raise ValueError(f"spacing must be finite, got {spacing!r}")
    if value < 0.0:
        raise ValueError(f"spacing must be >= 0, got {spacing!r}")
    return 1.0 + value / 2.0


def apply_spacing(points: np.ndarray, spacing: float) -> np.ndarray:
    """Dilate positions to realise a requested spacing.

    Parameters
    ----------
    points : (N, D) array
        Particle centres.
    spacing : float
        Clearance in units of the mean radius; see
        :func:`spacing_scale`.

    Returns
    -------
    numpy.ndarray
        A new ``(N, D)`` array of scaled positions.  The input is not
        modified and radii are unchanged by construction.

    See Also
    --------
    spacing_scale : The scalar factor applied here.
    expected_gap : The surface gap this opens for a given pair.
    """
    return np.asarray(points, dtype=np.float64) * spacing_scale(spacing)


def expected_gap(spacing: float, mean_radius: float) -> float:
    """Surface gap opened between two touching average particles.

    Parameters
    ----------
    spacing : float
        Clearance in units of the mean radius.
    mean_radius : float
        Arithmetic mean radius of the packing.

    Returns
    -------
    float
        ``spacing * mean_radius``.

    Notes
    -----
    This is the gap for a pair that was *touching* before scaling.
    Poisson-disc neighbours are generally not touching, so the realised
    minimum gap in a packing is larger; use
    :func:`polydisc.diagnostics.gap_report` for what was actually
    achieved.
    """
    return float(spacing) * float(mean_radius)
