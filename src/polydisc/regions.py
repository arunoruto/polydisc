"""Shapes that can be filled with particles.

A region is the container: it says where particles may sit, how much
volume there is to fill, and how large a particle can possibly fit.
Two are supported, :class:`Box` and :class:`Ball`, and both are placed
with their axis-aligned bounding box starting at the origin so the
acceleration grid can index them the same way.

A "carpet" is not a separate shape -- it is a :class:`Box` whose third
edge is close to a particle diameter.  See the README for the recipe.

A :class:`Box` may be periodic along any of its axes, so the packing is
one cell of an infinite (or laterally infinite) tiling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Kernel codes for the supported region shapes.
KIND_BOX = 0
KIND_BALL = 1


@dataclass(frozen=True, eq=False)
class Box:
    """An axis-aligned box with one corner at the origin.

    Parameters
    ----------
    bounds : array_like
        Edge lengths; two entries for 2D, three for 3D.  All must be
        finite and positive.
    periodic : bool or sequence of bool, default False
        Which axes wrap around.  A single bool applies to every axis.
        Along a periodic axis a particle leaving one face re-enters
        through the opposite one and distances follow the minimum-image
        convention, so copies of the box tile space without overlaps.
        The other axes keep hard walls that particles stay inside.

    Attributes
    ----------
    bounds : numpy.ndarray
        The validated edge lengths.
    periodic : tuple of bool
        One flag per axis.

    Raises
    ------
    ValueError
        If `bounds` is not 2 or 3 finite, positive numbers, or
        `periodic` does not have one entry per axis.

    Examples
    --------
    >>> box = Box([100.0, 50.0])
    >>> box.dimensions, box.volume
    (2, 5000.0)
    >>> box.max_particle_radius
    25.0

    A slab that is periodic sideways, with hard top and bottom faces:

    >>> slab = Box([32.0, 32.0, 44.0], periodic=(True, True, False))
    >>> slab.label
    'box, 32 x 32 x 44, periodic x, y'
    """

    bounds: np.ndarray
    periodic: Any = False

    def __post_init__(self) -> None:
        arr = np.asarray(self.bounds, dtype=np.float64).ravel()
        if arr.shape[0] not in (2, 3):
            raise ValueError(
                f"bounds must have 2 or 3 entries (2D or 3D), got {arr.shape[0]}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("bounds must all be finite")
        if np.any(arr <= 0.0):
            raise ValueError(f"bounds must all be > 0, got {arr.tolist()}")
        flags = self.periodic
        if isinstance(flags, (bool, np.bool_)):
            flags = (bool(flags),) * arr.shape[0]
        flags = tuple(bool(f) for f in flags)
        if len(flags) != arr.shape[0]:
            raise ValueError(
                f"periodic needs one flag per axis ({arr.shape[0]}), got {len(flags)}"
            )
        object.__setattr__(self, "bounds", arr)
        object.__setattr__(self, "periodic", flags)

    kind = KIND_BOX

    def __eq__(self, other: object) -> bool:
        """Compare edge lengths elementwise.

        The generated dataclass ``__eq__`` would compare the NumPy
        arrays with ``==`` and raise on the ambiguous truth value.
        """
        if not isinstance(other, Box):
            return NotImplemented
        return (
            bool(np.array_equal(self.bounds, other.bounds))
            and self.periodic == other.periodic
        )

    def __hash__(self) -> int:
        return hash((tuple(self.bounds.tolist()), self.periodic))

    @property
    def dimensions(self) -> int:
        """Spatial dimension of the region (2 or 3)."""
        return int(self.bounds.shape[0])

    @property
    def bounding_box(self) -> np.ndarray:
        """Edge lengths of the axis-aligned bounding box."""
        return self.bounds.copy()

    @property
    def volume(self) -> float:
        """Enclosed area (2D) or volume (3D)."""
        return float(np.prod(self.bounds))

    @property
    def footprint(self) -> float:
        """Area of the projection onto the first two axes."""
        return float(self.bounds[0] * self.bounds[1])

    @property
    def max_particle_radius(self) -> float:
        """Largest radius that can fit, ignoring other particles.

        A hard axis must hold one diameter; a periodic axis two, so that
        the minimum image of a neighbour is unambiguous.
        """
        per = np.asarray(self.periodic)
        return float(np.min(np.where(per, self.bounds / 2.0, self.bounds))) / 2.0

    def scaled(self, factor: float) -> Box:
        """Return a copy with every edge multiplied by `factor`.

        Parameters
        ----------
        factor : float
            Positive scale factor.

        Returns
        -------
        Box
            The scaled region.
        """
        return Box(self.bounds * float(factor), self.periodic)

    def contains(self, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Test which particles lie wholly inside the region.

        Parameters
        ----------
        points : (N, D) array_like
            Particle centres.
        radii : (N,) array_like
            Particle radii.

        Returns
        -------
        numpy.ndarray
            Boolean ``(N,)`` mask, true where the particle fits.  Along
            a periodic axis only the centre has to lie in the box.
        """
        pts = np.asarray(points, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64).reshape(-1, 1)
        per = np.asarray(self.periodic)[None, :]
        hi = self.bounds[None, :]
        inside = np.where(
            per, (pts >= 0.0) & (pts < hi), (pts >= rad) & (pts <= hi - rad)
        )
        return np.all(inside, axis=1)

    @property
    def label(self) -> str:
        """One-line description, e.g. ``"box, 400 x 400"``."""
        text = "box, " + " x ".join(f"{b:g}" for b in self.bounds)
        if any(self.periodic):
            axes = ", ".join("xyz"[i] for i, p in enumerate(self.periodic) if p)
            text += f", periodic {axes}"
        return text

    def admissible_measure(self, radii: np.ndarray) -> np.ndarray:
        """Measure of the sub-region a particle's centre may occupy.

        Parameters
        ----------
        radii : array_like
            Candidate particle radii.

        Returns
        -------
        numpy.ndarray
            For each radius, the area/volume of the shrunken box
            ``bounds - 2 * r``, zero where the particle cannot fit.
            Periodic axes are not shrunk.  Only relative values matter
            to callers.

        See Also
        --------
        polydisc.sampling.geometric_bias : Uses this to predict how the
            container skews the realised size distribution.
        """
        rad = np.asarray(radii, dtype=np.float64).ravel()
        room = np.clip(self.bounds[None, :] - 2.0 * rad[:, None], 0.0, None)
        room = np.where(np.asarray(self.periodic)[None, :], self.bounds[None, :], room)
        return np.prod(room, axis=1)

    def describe(self) -> dict[str, Any]:
        """Summarise the region as a JSON-friendly mapping."""
        out: dict[str, Any] = {"shape": "box", "bounds": self.bounds.tolist()}
        if any(self.periodic):
            out["periodic"] = list(self.periodic)
        return out


@dataclass(frozen=True)
class Ball:
    """A solid sphere (a ball) to be filled with particles.

    Named ``Ball`` rather than ``Sphere`` because the *particles* are
    spheres too; this is the container they go inside.  It is centred so
    that its bounding box runs from the origin to ``2 * radius`` on every
    axis, matching :class:`Box`.

    Parameters
    ----------
    radius : float
        Radius of the container.  Must be finite and positive.
    dimensions : int, default 3
        2 for a filled disc, 3 for a filled sphere.

    Attributes
    ----------
    radius : float
        The validated radius.
    dimensions : int
        2 or 3.

    Raises
    ------
    ValueError
        If `radius` is not finite and positive, or `dimensions` is not
        2 or 3.

    Examples
    --------
    >>> ball = Ball(10.0)
    >>> ball.bounding_box.tolist()
    [20.0, 20.0, 20.0]
    >>> ball.max_particle_radius
    10.0
    >>> round(Ball(1.0, dimensions=2).volume, 6)
    3.141593
    """

    radius: float
    dimensions: int = 3

    def __post_init__(self) -> None:
        value = float(self.radius)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"radius must be a finite value > 0, got {self.radius!r}")
        if int(self.dimensions) not in (2, 3):
            raise ValueError(f"dimensions must be 2 or 3, got {self.dimensions!r}")
        object.__setattr__(self, "radius", value)
        object.__setattr__(self, "dimensions", int(self.dimensions))

    kind = KIND_BALL

    @property
    def centre(self) -> np.ndarray:
        """Centre of the ball in the region's own coordinates."""
        return np.full(self.dimensions, self.radius, dtype=np.float64)

    @property
    def bounding_box(self) -> np.ndarray:
        """Edge lengths of the axis-aligned bounding box."""
        return np.full(self.dimensions, 2.0 * self.radius, dtype=np.float64)

    @property
    def volume(self) -> float:
        """Enclosed area (2D) or volume (3D)."""
        if self.dimensions == 2:
            return math.pi * self.radius**2
        return (4.0 / 3.0) * math.pi * self.radius**3

    @property
    def footprint(self) -> float:
        """Area of the projection onto the first two axes."""
        return math.pi * self.radius**2

    @property
    def max_particle_radius(self) -> float:
        """Largest radius that can fit, ignoring other particles."""
        return self.radius

    def scaled(self, factor: float) -> Ball:
        """Return a copy with the radius multiplied by `factor`.

        Parameters
        ----------
        factor : float
            Positive scale factor.

        Returns
        -------
        Ball
            The scaled region.
        """
        return Ball(self.radius * float(factor), self.dimensions)

    def contains(self, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Test which particles lie wholly inside the region.

        Parameters
        ----------
        points : (N, D) array_like
            Particle centres, in the region's own coordinates.
        radii : (N,) array_like
            Particle radii.

        Returns
        -------
        numpy.ndarray
            Boolean ``(N,)`` mask, true where the particle fits.
        """
        pts = np.asarray(points, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64).ravel()
        distance = np.linalg.norm(pts - self.centre[None, :], axis=1)
        return distance <= self.radius - rad

    @property
    def label(self) -> str:
        """One-line description, e.g. ``"ball, radius 200"``."""
        return f"ball, radius {self.radius:g}"

    def admissible_measure(self, radii: np.ndarray) -> np.ndarray:
        """Measure of the sub-region a particle's centre may occupy.

        Parameters
        ----------
        radii : array_like
            Candidate particle radii.

        Returns
        -------
        numpy.ndarray
            For each radius, the measure of the ball of radius
            ``self.radius - r``, zero where the particle cannot fit.
            Only relative values matter to callers.

        Notes
        -----
        The *relative* falloff, ``(1 - r / R) ** d``, is identical to
        that of the ball's own bounding cube, so swapping a cube for the
        ball inscribed in it does not change the size bias at all.  It
        is milder than that of an equal-*volume* box, whose inradius is
        smaller than ``R``.
        """
        rad = np.asarray(radii, dtype=np.float64).ravel()
        room = np.clip(self.radius - rad, 0.0, None)
        return room**self.dimensions

    def describe(self) -> dict[str, Any]:
        """Summarise the region as a JSON-friendly mapping."""
        return {
            "shape": "ball",
            "radius": self.radius,
            "dimensions": self.dimensions,
        }


#: Anything the samplers accept as a container.
Region = Box | Ball


def as_region(region: Region | Sequence[float] | np.ndarray) -> Region:
    """Coerce a user-supplied container argument into a region.

    Parameters
    ----------
    region : Box or Ball or array_like
        An existing region, returned unchanged, or a sequence of edge
        lengths read as a :class:`Box`.

    Returns
    -------
    Box or Ball
        The coerced region.

    Raises
    ------
    TypeError
        If `region` is a type that cannot be interpreted.

    Examples
    --------
    >>> as_region([10.0, 20.0]).describe()["shape"]
    'box'
    """
    if isinstance(region, (Box, Ball)):
        return region
    if isinstance(region, (np.ndarray, list, tuple)):
        return Box(region)
    raise TypeError(
        "region must be a Box, a Ball, or a sequence of box edge "
        f"lengths; got {type(region)!r}"
    )


def region_from_metadata(meta: dict[str, Any] | None) -> Region | None:
    """Rebuild a region from its :meth:`describe` output.

    Parameters
    ----------
    meta : dict or None
        A mapping produced by ``Box.describe`` or ``Ball.describe``.

    Returns
    -------
    Box or Ball or None
        The restored region, or ``None`` if `meta` is unusable.
    """
    if not isinstance(meta, dict):
        return None
    try:
        if meta.get("shape") == "box":
            return Box(meta["bounds"], meta.get("periodic", False))
        if meta.get("shape") == "ball":
            return Ball(meta["radius"], meta.get("dimensions", 3))
    except (KeyError, ValueError, TypeError):
        return None
    return None
