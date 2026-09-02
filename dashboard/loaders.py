"""Turn an uploaded file into a :class:`polydisc.Packing`.

Two formats round-trip from the CLI: ``.npz``, which carries the full
run metadata, and ``.csv``, which carries only coordinates and radii.
For CSV -- and for an ``.npz`` written by something else -- the
container and the size distribution have to be inferred from the data,
which this module does explicitly rather than pretending the
information was there.
"""

from __future__ import annotations

import csv
import io

import numpy as np

from polydisc.radii import RadiusSpec
from polydisc.regions import Box
from polydisc.sampling import Packing

#: Column headers accepted in a CSV, in order.
CSV_HEADERS_2D = ("x", "y", "r")
CSV_HEADERS_3D = ("x", "y", "z", "r")

#: Distinct radii above which an inferred distribution is treated as
#: continuous (uniform over the observed range) rather than a discrete
#: set.  A generated packing has as many distinct radii as particles
#: when the spec was continuous, and a handful when it was discrete.
_DISCRETE_LIMIT = 24


class LoadError(ValueError):
    """Raised when an uploaded file cannot be read as a packing."""


def infer_region(points: np.ndarray, radii: np.ndarray) -> Box:
    """Infer the smallest box that contains every particle.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.

    Returns
    -------
    Box
        A box spanning the particles' full extent, with the origin at
        the low corner.  The points are *not* shifted; call
        :func:`shift_to_origin` for that.

    Examples
    --------
    >>> points = np.array([[1.0, 1.0], [4.0, 3.0]])
    >>> infer_region(points, np.array([1.0, 1.0])).bounds.tolist()
    [5.0, 4.0]
    """
    low = np.min(points - radii[:, None], axis=0)
    high = np.max(points + radii[:, None], axis=0)
    return Box(np.maximum(high - low, np.finfo(np.float64).tiny))


def shift_to_origin(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Translate particles so the packing's low corner sits at zero.

    Parameters
    ----------
    points : (N, D) numpy.ndarray
        Particle centres.
    radii : (N,) numpy.ndarray
        Particle radii.

    Returns
    -------
    numpy.ndarray
        A new ``(N, D)`` array of shifted centres.
    """
    low = np.min(points - radii[:, None], axis=0)
    return points - low[None, :]


def infer_spec(radii: np.ndarray) -> RadiusSpec:
    """Infer a plausible radius distribution from realised radii.

    Parameters
    ----------
    radii : (N,) numpy.ndarray
        Realised particle radii.

    Returns
    -------
    RadiusSpec
        A discrete spec weighted by observed frequency when few
        distinct radii are present, otherwise a uniform spec over the
        observed range.

    Notes
    -----
    This is a description of the file, not of whatever distribution
    actually produced it, so the "requested vs realised" comparison in
    the report is meaningless for an inferred spec and the dashboard
    says so.
    """
    values, counts = np.unique(radii, return_counts=True)
    if values.size <= _DISCRETE_LIMIT:
        return RadiusSpec.discrete(values, counts.astype(np.float64))
    return RadiusSpec.uniform(float(values[0]), float(values[-1]))


def _packing_from_arrays(
    points: np.ndarray, radii: np.ndarray, *, source: str
) -> Packing:
    """Assemble a packing from bare coordinates, inferring the rest."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    radii = np.ascontiguousarray(radii, dtype=np.float64).ravel()

    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise LoadError(f"{source}: coordinates must be 2D or 3D, got {points.shape}")
    if radii.shape[0] != points.shape[0]:
        raise LoadError(
            f"{source}: {radii.shape[0]} radii for {points.shape[0]} points"
        )
    if points.shape[0] == 0:
        raise LoadError(f"{source}: no particles")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(radii)):
        raise LoadError(f"{source}: coordinates and radii must all be finite")
    if np.any(radii <= 0.0):
        raise LoadError(f"{source}: radii must all be > 0")

    shifted = shift_to_origin(points, radii)
    return Packing(
        points=shifted,
        radii=radii,
        region=infer_region(points, radii),
        spec=infer_spec(radii),
    )


def packing_from_csv(data: bytes) -> Packing:
    """Read a packing from CSV bytes.

    Parameters
    ----------
    data : bytes
        Contents of a ``x,y,r`` or ``x,y,z,r`` CSV, as written by
        ``polydisc generate``.

    Returns
    -------
    Packing
        The particles, with the container and size distribution
        inferred from the data.

    Raises
    ------
    LoadError
        If the header is missing or unrecognised, or a row is malformed.

    Examples
    --------
    >>> pack = packing_from_csv(b"x,y,r\\n1,1,1\\n4,1,1\\n")
    >>> pack.n, pack.dimensions
    (2, 2)
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LoadError("CSV: file is not UTF-8 text") from exc

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise LoadError("CSV: file is empty") from exc

    columns = tuple(name.strip().lower() for name in header)
    if columns not in (CSV_HEADERS_2D, CSV_HEADERS_3D):
        raise LoadError(
            f"CSV: expected a header of 'x,y,r' or 'x,y,z,r', got {','.join(columns)!r}"
        )

    rows: list[list[float]] = []
    for number, row in enumerate(reader, start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(columns):
            raise LoadError(
                f"CSV line {number}: expected {len(columns)} values, got {len(row)}"
            )
        try:
            rows.append([float(cell) for cell in row])
        except ValueError as exc:
            raise LoadError(f"CSV line {number}: {exc}") from exc

    if not rows:
        raise LoadError("CSV: no data rows")

    table = np.asarray(rows, dtype=np.float64)
    return _packing_from_arrays(table[:, :-1], table[:, -1], source="CSV")


def packing_from_npz(data: bytes) -> Packing:
    """Read a packing from ``.npz`` bytes.

    Parameters
    ----------
    data : bytes
        Contents of an archive written by :meth:`polydisc.Packing.save`,
        or any archive holding ``points`` and ``radii`` arrays.

    Returns
    -------
    Packing
        The restored packing.  Full metadata is used when present;
        otherwise the container and size distribution are inferred.

    Raises
    ------
    LoadError
        If the archive cannot be read or lacks ``points``/``radii``.
    """
    # Every .npz is a zip archive; without this, NumPy reports a
    # confusing message about pickles for, say, an uploaded image.
    if not data.startswith(b"PK"):
        raise LoadError("NPZ: this is not a .npz archive")

    try:
        return Packing.load(io.BytesIO(data))
    except (KeyError, ValueError, OSError):
        # Not one of ours, or missing the metadata: fall back to
        # whatever arrays it does carry.
        pass

    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            names = set(archive.files)
            if not {"points", "radii"} <= names:
                raise LoadError(
                    "NPZ: expected arrays named 'points' and 'radii', found "
                    + (", ".join(sorted(names)) or "none")
                )
            points = np.asarray(archive["points"])
            radii = np.asarray(archive["radii"])
    except LoadError:
        raise
    except (ValueError, OSError) as exc:
        raise LoadError(f"NPZ: could not read the archive ({exc})") from exc

    return _packing_from_arrays(points, radii, source="NPZ")


def load_packing(filename: str, data: bytes) -> Packing:
    """Read a packing from an uploaded file, dispatching on suffix.

    Parameters
    ----------
    filename : str
        Name of the uploaded file; only its suffix is used.
    data : bytes
        The file's contents.

    Returns
    -------
    Packing
        The loaded packing.

    Raises
    ------
    LoadError
        If the suffix is not ``.npz`` or ``.csv``, or the contents
        cannot be read.

    Examples
    --------
    >>> load_packing("pack.csv", b"x,y,r\\n0,0,1\\n3,0,1\\n").n
    2
    """
    lowered = filename.lower()
    if lowered.endswith(".npz"):
        return packing_from_npz(data)
    if lowered.endswith(".csv"):
        return packing_from_csv(data)
    raise LoadError(f"{filename}: expected a .npz or .csv file")
