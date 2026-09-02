"""Radius distributions for polydisperse sampling.

Two families are supported: a *discrete* set of allowed radii with
optional weights, and *continuous* distributions (lognormal, normal,
uniform) truncated to a finite ``[rmin, rmax]`` window.  Both are
reduced to a small plain-data record (:class:`RadiusSpec`) that the
Numba kernels can consume without any Python objects.

The lognormal parameterisation follows the aerosol/particle-science
convention also used by PyFracVAL: a geometric mean radius ``gmean``
and a geometric standard deviation ``gstd >= 1``, truncated by default
at +/- 2 GSD, i.e. ``[gmean / gstd**2, gmean * gstd**2]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

#: Kernel codes for the supported distribution families.  These are
#: passed to the Numba kernels as plain integers.
KIND_DISCRETE = 0
KIND_LOGNORMAL = 1
KIND_NORMAL = 2
KIND_UNIFORM = 3

_KIND_NAMES = {
    KIND_DISCRETE: "discrete",
    KIND_LOGNORMAL: "lognormal",
    KIND_NORMAL: "normal",
    KIND_UNIFORM: "uniform",
}

DistName = Literal["discrete", "lognormal", "normal", "uniform"]


@dataclass(frozen=True)
class RadiusSpec:
    """A radius distribution reduced to kernel-ready plain data.

    Instances are normally built through :meth:`discrete`,
    :meth:`lognormal`, :meth:`normal`, :meth:`uniform` or
    :func:`as_radius_spec` rather than constructed directly.

    Parameters
    ----------
    kind : int
        One of the ``KIND_*`` constants in this module.
    values : numpy.ndarray
        Allowed radii for a discrete spec; empty for continuous ones.
    cdf : numpy.ndarray
        Cumulative weights matching `values`; empty for continuous ones.
    p0, p1 : float
        Distribution parameters.  ``(mu, sigma)`` of the underlying
        normal for ``lognormal``, ``(mean, std)`` for ``normal``, and
        unused for ``uniform`` and ``discrete``.
    rmin, rmax : float
        Support of the distribution.  Continuous draws outside this
        window are rejected; `rmax` also sets the neighbour-search cell
        size, so it must be a true upper bound.

    Attributes
    ----------
    name : str
        Human-readable distribution family name.
    """

    kind: int
    values: np.ndarray
    cdf: np.ndarray
    p0: float
    p1: float
    rmin: float
    rmax: float

    @property
    def name(self) -> str:
        """Human-readable distribution family name."""
        return _KIND_NAMES[self.kind]

    # -- constructors ---------------------------------------------------

    @classmethod
    def discrete(
        cls,
        values: Sequence[float] | np.ndarray,
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> RadiusSpec:
        """Build a spec drawing from a finite set of radii.

        Parameters
        ----------
        values : array_like
            Allowed radii.  Must be finite and strictly positive.
        weights : array_like, optional
            Relative probabilities, same length as `values`.  Defaults
            to uniform.  Normalised internally; need not sum to one.

        Returns
        -------
        RadiusSpec
            A spec with ``kind == KIND_DISCRETE``.

        Raises
        ------
        ValueError
            If `values` is empty, non-positive or non-finite, or if
            `weights` has a mismatched shape, negative entries or a
            non-positive sum.

        Examples
        --------
        >>> spec = RadiusSpec.discrete([1.0, 2.0], weights=[3.0, 1.0])
        >>> spec.rmin, spec.rmax
        (1.0, 2.0)
        """
        vals = np.asarray(values, dtype=np.float64).ravel()
        if vals.size == 0:
            raise ValueError("radii must contain at least one value")
        if not np.all(np.isfinite(vals)):
            raise ValueError("radii must all be finite")
        if np.any(vals <= 0.0):
            raise ValueError("radii must all be > 0")

        if weights is None:
            w = np.full(vals.shape[0], 1.0, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.shape != vals.shape:
                raise ValueError(
                    f"weights shape {w.shape} does not match radii shape {vals.shape}"
                )
            if not np.all(np.isfinite(w)):
                raise ValueError("weights must all be finite")
            if np.any(w < 0.0):
                raise ValueError("weights must be non-negative")

        total = float(np.sum(w))
        if total <= 0.0:
            raise ValueError("weights must sum to > 0")

        cdf = np.cumsum(w / total)
        cdf[-1] = 1.0
        return cls(
            kind=KIND_DISCRETE,
            values=vals,
            cdf=cdf,
            p0=0.0,
            p1=0.0,
            rmin=float(np.min(vals)),
            rmax=float(np.max(vals)),
        )

    @classmethod
    def lognormal(
        cls,
        gmean: float,
        gstd: float,
        *,
        truncate: float = 2.0,
        rmin: float | None = None,
        rmax: float | None = None,
    ) -> RadiusSpec:
        """Build a truncated lognormal radius spec.

        Parameters
        ----------
        gmean : float
            Geometric mean (median) radius.  Must be ``> 0``.
        gstd : float
            Geometric standard deviation, ``>= 1``.  A value of exactly
            ``1`` yields a monodisperse spec.
        truncate : float, optional
            Truncation half-width in geometric standard deviations.
            The default of ``2`` gives the window
            ``[gmean / gstd**2, gmean * gstd**2]`` used by FracVAL and
            PyFracVAL.  Use ``0`` or :data:`math.inf` to disable, in
            which case `rmin` and `rmax` must be given.
        rmin, rmax : float, optional
            Explicit truncation bounds, overriding `truncate`.

        Returns
        -------
        RadiusSpec
            A spec with ``kind == KIND_LOGNORMAL``.

        Raises
        ------
        ValueError
            If `gmean` is not positive, `gstd` is below one, the
            resulting window is empty, or an unbounded distribution is
            requested without explicit bounds.

        Notes
        -----
        Truncation preserves the geometric mean but shrinks every
        arithmetic moment; the realised geometric standard deviation of
        a +/- 2 GSD window is about ``gstd ** 0.88``.  Use
        :func:`polydisc.diagnostics.radius_report` to see what was
        actually realised.

        Examples
        --------
        >>> spec = RadiusSpec.lognormal(10.0, 1.5)
        >>> round(spec.rmin, 4), round(spec.rmax, 4)
        (4.4444, 22.5)
        """
        if not math.isfinite(gmean) or gmean <= 0.0:
            raise ValueError(f"gmean must be a finite value > 0, got {gmean!r}")
        if not math.isfinite(gstd) or gstd < 1.0:
            raise ValueError(f"gstd must be a finite value >= 1, got {gstd!r}")

        if gstd == 1.0:
            return cls.discrete([gmean])

        mu = math.log(gmean)
        sigma = math.log(gstd)

        if rmin is None or rmax is None:
            if not math.isfinite(truncate) or truncate <= 0.0:
                raise ValueError(
                    "an untruncated lognormal has no finite rmax; pass "
                    "truncate > 0, or explicit rmin and rmax"
                )
            auto_lo = gmean * gstd ** (-truncate)
            auto_hi = gmean * gstd**truncate
            rmin = auto_lo if rmin is None else rmin
            rmax = auto_hi if rmax is None else rmax

        return cls._continuous(KIND_LOGNORMAL, mu, sigma, rmin, rmax)

    @classmethod
    def normal(
        cls,
        mean: float,
        std: float,
        *,
        truncate: float = 2.0,
        rmin: float | None = None,
        rmax: float | None = None,
    ) -> RadiusSpec:
        """Build a truncated normal radius spec.

        Parameters
        ----------
        mean : float
            Arithmetic mean radius, ``> 0``.
        std : float
            Arithmetic standard deviation, ``>= 0``.
        truncate : float, optional
            Truncation half-width in standard deviations.  Default 2.
        rmin, rmax : float, optional
            Explicit truncation bounds, overriding `truncate`.

        Returns
        -------
        RadiusSpec
            A spec with ``kind == KIND_NORMAL``.

        Raises
        ------
        ValueError
            If `mean` is not positive, `std` is negative, or the
            resulting window is empty or reaches non-positive radii.
        """
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError(f"mean must be a finite value > 0, got {mean!r}")
        if not math.isfinite(std) or std < 0.0:
            raise ValueError(f"std must be a finite value >= 0, got {std!r}")

        if std == 0.0:
            return cls.discrete([mean])

        if rmin is None or rmax is None:
            if not math.isfinite(truncate) or truncate <= 0.0:
                raise ValueError(
                    "an untruncated normal has no finite rmax; pass "
                    "truncate > 0, or explicit rmin and rmax"
                )
            rmin = mean - truncate * std if rmin is None else rmin
            rmax = mean + truncate * std if rmax is None else rmax

        return cls._continuous(KIND_NORMAL, mean, std, rmin, rmax)

    @classmethod
    def uniform(cls, rmin: float, rmax: float) -> RadiusSpec:
        """Build a spec drawing radii uniformly from ``[rmin, rmax]``.

        Parameters
        ----------
        rmin, rmax : float
            Inclusive bounds of the radius range.  ``0 < rmin <= rmax``.

        Returns
        -------
        RadiusSpec
            A spec with ``kind == KIND_UNIFORM``.

        Raises
        ------
        ValueError
            If the bounds are non-finite, non-positive or inverted.
        """
        return cls._continuous(KIND_UNIFORM, 0.0, 0.0, rmin, rmax)

    @classmethod
    def _continuous(
        cls, kind: int, p0: float, p1: float, rmin: float, rmax: float
    ) -> RadiusSpec:
        """Validate and package a continuous distribution."""
        rmin = float(rmin)
        rmax = float(rmax)
        if not (math.isfinite(rmin) and math.isfinite(rmax)):
            raise ValueError("rmin and rmax must both be finite")
        if rmin <= 0.0:
            raise ValueError(f"rmin must be > 0, got {rmin!r}")
        if rmax < rmin:
            raise ValueError(f"rmax ({rmax!r}) must be >= rmin ({rmin!r})")

        empty = np.empty(0, dtype=np.float64)
        return cls(
            kind=kind,
            values=empty,
            cdf=empty,
            p0=float(p0),
            p1=float(p1),
            rmin=rmin,
            rmax=rmax,
        )

    # -- drawing --------------------------------------------------------

    def sample(self, size: int, seed: int | None = None) -> np.ndarray:
        """Draw radii from the distribution.

        Parameters
        ----------
        size : int
            Number of radii to draw.  Must be ``>= 0``.
        seed : int, optional
            Seed for the kernel's generator.  ``None`` draws one from
            NumPy's global generator.

        Returns
        -------
        numpy.ndarray
            ``(size,)`` array of radii, all inside ``[rmin, rmax]``.

        Raises
        ------
        ValueError
            If `size` is negative.

        Notes
        -----
        This is the very routine the samplers call per candidate, so
        the distribution seen here is the one they draw from — before
        any placement bias (see :mod:`polydisc.diagnostics`).

        Examples
        --------
        >>> spec = RadiusSpec.uniform(1.0, 2.0)
        >>> draws = spec.sample(100, seed=0)
        >>> bool(draws.min() >= 1.0 and draws.max() <= 2.0)
        True
        """
        from polydisc._kernels import _draw_radii

        count = int(size)
        if count < 0:
            raise ValueError(f"size must be >= 0, got {size!r}")
        if seed is None:
            seed_int = int(np.random.randint(0, 2**31 - 1))
        else:
            seed_int = int(seed) % (2**31 - 1)
        return _draw_radii(
            count,
            self.kind,
            self.values,
            self.cdf,
            self.p0,
            self.p1,
            self.rmin,
            self.rmax,
            seed_int,
        )

    # -- reporting ------------------------------------------------------

    def cdf_at(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the (truncated) cumulative distribution function.

        Parameters
        ----------
        r : numpy.ndarray
            Radii at which to evaluate the CDF.

        Returns
        -------
        numpy.ndarray
            Values in ``[0, 1]``, the same shape as `r`.  For a
            discrete spec this is the right-continuous step CDF.
        """
        r = np.asarray(r, dtype=np.float64)
        if self.kind == KIND_DISCRETE:
            order = np.argsort(self.values)
            sorted_vals = self.values[order]
            pmf = np.diff(np.concatenate(([0.0], self.cdf)))[order]
            sorted_cdf = np.cumsum(pmf)
            idx = np.searchsorted(sorted_vals, r, side="right") - 1
            out = np.where(idx < 0, 0.0, sorted_cdf[np.clip(idx, 0, None)])
            return np.asarray(out, dtype=np.float64)

        if self.kind == KIND_UNIFORM:
            raw = (r - self.rmin) / max(self.rmax - self.rmin, 1e-300)
            return np.clip(raw, 0.0, 1.0)

        if self.kind == KIND_LOGNORMAL:
            z = (np.log(np.maximum(r, 1e-300)) - self.p0) / self.p1
            lo = (math.log(self.rmin) - self.p0) / self.p1
            hi = (math.log(self.rmax) - self.p0) / self.p1
        else:  # KIND_NORMAL
            z = (r - self.p0) / self.p1
            lo = (self.rmin - self.p0) / self.p1
            hi = (self.rmax - self.p0) / self.p1

        norm = _phi(hi) - _phi(lo)
        if norm <= 0.0:
            return np.clip((r - self.rmin) / max(self.rmax - self.rmin, 1e-300), 0, 1)
        return np.clip((_phi(z) - _phi(lo)) / norm, 0.0, 1.0)

    def discretize(self, samples: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Represent the distribution as values with probabilities.

        Parameters
        ----------
        samples : int, optional
            Number of bins used for a continuous spec.  Ignored for a
            discrete one, which is already exact.  Default 512.

        Returns
        -------
        values : numpy.ndarray
            Radii; the spec's own values for a discrete spec, bin
            centres otherwise.
        pmf : numpy.ndarray
            Matching probabilities, summing to one.

        Examples
        --------
        >>> values, pmf = RadiusSpec.discrete([1.0, 2.0]).discretize()
        >>> values.tolist(), pmf.tolist()
        ([1.0, 2.0], [0.5, 0.5])
        """
        if self.kind == KIND_DISCRETE:
            pmf = np.diff(np.concatenate(([0.0], self.cdf)))
            return self.values, pmf

        if self.rmax <= self.rmin:
            return np.array([self.rmin]), np.array([1.0])

        edges = np.linspace(self.rmin, self.rmax, int(samples) + 1)
        weight = np.diff(self.cdf_at(edges))
        total = float(np.sum(weight))
        centres = 0.5 * (edges[:-1] + edges[1:])
        if total <= 0.0:
            return centres, np.full(centres.shape, 1.0 / centres.shape[0])
        return centres, weight / total

    def describe(self) -> dict[str, Any]:
        """Summarise the spec as a JSON-friendly mapping.

        Returns
        -------
        dict
            Keys always include ``"kind"``, ``"rmin"`` and ``"rmax"``;
            discrete specs add ``"values"`` and ``"weights"``, while
            continuous specs add the family's parameters.
        """
        out: dict[str, Any] = {
            "kind": self.name,
            "rmin": self.rmin,
            "rmax": self.rmax,
        }
        if self.kind == KIND_DISCRETE:
            pmf = np.diff(np.concatenate(([0.0], self.cdf)))
            out["values"] = self.values.tolist()
            out["weights"] = pmf.tolist()
        elif self.kind == KIND_LOGNORMAL:
            out["gmean"] = math.exp(self.p0)
            out["gstd"] = math.exp(self.p1)
        elif self.kind == KIND_NORMAL:
            out["mean"] = self.p0
            out["std"] = self.p1
        return out


def _phi(x: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF, vectorised, without a SciPy dependency."""
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def as_radius_spec(
    radii: RadiusSpec | Sequence[float] | np.ndarray | float,
    weights: Sequence[float] | np.ndarray | None = None,
) -> RadiusSpec:
    """Coerce a user-supplied radius argument into a :class:`RadiusSpec`.

    Parameters
    ----------
    radii : RadiusSpec or array_like or float
        An existing spec (returned unchanged), a scalar radius, or a
        sequence of allowed radii.
    weights : array_like, optional
        Weights for the discrete case.  Rejected when `radii` is
        already a :class:`RadiusSpec`.

    Returns
    -------
    RadiusSpec
        The coerced spec.

    Raises
    ------
    TypeError
        If `radii` is a type that cannot be interpreted.
    ValueError
        If `weights` is combined with an existing :class:`RadiusSpec`.

    Examples
    --------
    >>> as_radius_spec(3.0).values.tolist()
    [3.0]
    """
    if isinstance(radii, RadiusSpec):
        if weights is not None:
            raise ValueError(
                "weights cannot be combined with an existing RadiusSpec; "
                "pass them to RadiusSpec.discrete() instead"
            )
        return radii

    if isinstance(radii, (int, float, np.floating, np.integer)):
        return RadiusSpec.discrete([float(radii)], weights)

    if isinstance(radii, (np.ndarray, list, tuple)):
        return RadiusSpec.discrete(radii, weights)

    if callable(radii):
        raise TypeError(
            "callable radii are no longer supported; use "
            "RadiusSpec.lognormal/normal/uniform for continuous "
            "distributions, or RadiusSpec.discrete for a fixed set"
        )

    raise TypeError(
        "radii must be a RadiusSpec, a float, or a sequence of radii; "
        f"got {type(radii)!r}"
    )
