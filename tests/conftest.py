"""Shared fixtures and brute-force reference helpers for the test suite.

The helpers here deliberately re-derive properties with plain NumPy
(O(N^2) where needed) so they are independent of the accelerated code
paths they check.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """A seeded generator for tests that need auxiliary randomness."""
    return np.random.default_rng(20260902)


def brute_force_min_gaps(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Nearest-neighbour surface gaps, computed all-pairs.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` array of ``min_{j != i} (d_ij - r_i - r_j)``.
    """
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    gaps = dist - radii[:, None] - radii[None, :]
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def assert_no_overlaps(
    points: np.ndarray, radii: np.ndarray, *, tol: float = 1e-9
) -> None:
    """Fail if any two particles interpenetrate.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.
    tol : float, optional
        Relative slack allowed on the contact distance, absorbing the
        last-ulp difference between the kernel's squared-distance test
        and this recomputation.  Default ``1e-9``.
    """
    if len(points) < 2:
        return
    gaps = brute_force_min_gaps(points, radii)
    worst = float(np.min(gaps))
    allowed = -tol * float(np.mean(radii))
    assert worst >= allowed, (
        f"{int(np.sum(gaps < allowed))} particles overlap; "
        f"worst penetration {-worst:.3e}"
    )


def assert_inside(
    points: np.ndarray,
    radii: np.ndarray,
    bounds: np.ndarray,
    *,
    tol: float = 1e-9,
) -> None:
    """Fail if any particle pokes outside the domain.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.
    bounds : (D,) numpy.ndarray
        Domain edge lengths, with the domain starting at the origin.
    tol : float, optional
        Absolute slack, scaled by the mean radius.  Default ``1e-9``.
    """
    assert points.shape[0] == radii.shape[0]
    assert points.shape[1] == bounds.shape[0]
    if len(points) == 0:
        return
    slack = tol * float(np.mean(radii))
    assert np.all(points >= radii[:, None] - slack), "particle crosses the low face"
    assert np.all(points <= bounds[None, :] - radii[:, None] + slack), (
        "particle crosses the high face"
    )
