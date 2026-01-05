import numpy as np
from numba import njit


def _prepare_radii_cdf(
    allowed_radii: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    allowed_radii_arr = np.asarray(allowed_radii, dtype=np.float64)

    if weights is None:
        w = np.full(allowed_radii_arr.shape[0], 1.0, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != allowed_radii_arr.shape:
            raise ValueError("weights must have same shape as allowed_radii")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("weights must sum to > 0")

    w = w / total
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    return cdf


@njit(cache=True)
def _random_unit_vector(dimensions: int) -> np.ndarray:
    v = np.random.normal(0.0, 1.0, dimensions)
    norm = np.sqrt(np.sum(v * v))
    if norm == 0.0:
        v[0] = 1.0
        return v
    return v / norm


@njit(cache=True)
def _sample_from_cdf(values: np.ndarray, cdf: np.ndarray) -> float:
    u = np.random.random()

    lo = 0
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1

    return float(values[lo])


@njit(cache=True)
def _poisson_disc_samples_2d_numba(
    bounds: np.ndarray,
    allowed_radii: np.ndarray,
    radii_cdf: np.ndarray,
    repetitions: int,
    rmin: float,
    rmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimensions = bounds.shape[0]
    if dimensions != 2:
        raise ValueError("Expected 2D bounds")

    cell_size = 2.0 * rmin / np.sqrt(dimensions)
    cell_bounds = np.ceil(bounds / cell_size).astype(np.int64)

    background_grid = np.empty(
        (cell_bounds[0], cell_bounds[1], dimensions + 1), dtype=np.float64
    )
    background_grid[:] = np.nan

    # Step 1: initial sample
    radius = _sample_from_cdf(allowed_radii, radii_cdf)

    position = np.random.random(dimensions) * (bounds - 2.0 * radius) + radius

    position_index = np.floor(position / cell_size).astype(np.int64)
    background_grid[position_index[0], position_index[1], :dimensions] = position
    background_grid[position_index[0], position_index[1], dimensions] = radius

    active_points = np.empty(
        (cell_bounds[0] * cell_bounds[1], dimensions), dtype=np.float64
    )
    active_radii = np.empty(cell_bounds[0] * cell_bounds[1], dtype=np.float64)
    active_count = 1
    active_points[0] = position
    active_radii[0] = radius

    while active_count > 0:
        active_index = np.random.randint(active_count)
        active_position = active_points[active_index]
        active_radius = active_radii[active_index]

        found = False
        for _ in range(repetitions):
            test_radius = _sample_from_cdf(allowed_radii, radii_cdf)

            offset = _random_unit_vector(dimensions)

            # Bridson-style candidate generation in an annulus.
            # Here we approximate a variable-radius analogue by using
            # r_sum = (active_radius + test_radius) and sampling uniformly from
            # [r_sum, 2*r_sum].
            r_sum = active_radius + test_radius
            offset *= r_sum * (1.0 + np.random.random())
            test_position = active_position + offset

            # bounds check
            oob = False
            for d in range(dimensions):
                if (
                    test_position[d] < test_radius
                    or test_position[d] > bounds[d] - test_radius
                ):
                    oob = True
                    break
            if oob:
                continue

            # neighbourhood scan
            test_grid_radius = int(np.ceil((rmax + test_radius) / cell_size))
            cell_index = np.floor(test_position / cell_size).astype(np.int64)

            x0 = max(0, cell_index[0] - test_grid_radius)
            x1 = min(cell_bounds[0], cell_index[0] + test_grid_radius + 1)
            y0 = max(0, cell_index[1] - test_grid_radius)
            y1 = min(cell_bounds[1], cell_index[1] + test_grid_radius + 1)

            collision = False
            for gx in range(x0, x1):
                for gy in range(y0, y1):
                    r_other = background_grid[gx, gy, dimensions]
                    if np.isnan(r_other):
                        continue

                    dist2 = 0.0
                    for d in range(dimensions):
                        delta = test_position[d] - background_grid[gx, gy, d]
                        dist2 += delta * delta

                    min_dist = test_radius + r_other
                    if dist2 < min_dist * min_dist:
                        collision = True
                        break
                if collision:
                    break

            if collision:
                continue

            # accept point
            background_grid[cell_index[0], cell_index[1], :dimensions] = test_position
            background_grid[cell_index[0], cell_index[1], dimensions] = test_radius

            active_points[active_count] = test_position
            active_radii[active_count] = test_radius
            active_count += 1
            found = True
            break

        if not found:
            last = active_count - 1
            active_points[active_index] = active_points[last]
            active_radii[active_index] = active_radii[last]
            active_count -= 1

    # Extract samples from the grid
    total_cells = cell_bounds[0] * cell_bounds[1]
    tmp_points = np.empty((total_cells, dimensions), dtype=np.float64)
    tmp_radii = np.empty(total_cells, dtype=np.float64)
    out_count = 0

    for gx in range(cell_bounds[0]):
        for gy in range(cell_bounds[1]):
            r = background_grid[gx, gy, dimensions]
            if np.isnan(r):
                continue
            tmp_points[out_count] = background_grid[gx, gy, :dimensions]
            tmp_radii[out_count] = r
            out_count += 1

    return tmp_points[:out_count], tmp_radii[:out_count]


@njit(cache=True)
def _poisson_disc_samples_3d_numba(
    bounds: np.ndarray,
    allowed_radii: np.ndarray,
    radii_cdf: np.ndarray,
    repetitions: int,
    rmin: float,
    rmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimensions = bounds.shape[0]
    if dimensions != 3:
        raise ValueError("Expected 3D bounds")

    cell_size = 2.0 * rmin / np.sqrt(dimensions)
    cell_bounds = np.ceil(bounds / cell_size).astype(np.int64)

    background_grid = np.empty(
        (cell_bounds[0], cell_bounds[1], cell_bounds[2], dimensions + 1),
        dtype=np.float64,
    )
    background_grid[:] = np.nan

    # Step 1: initial sample
    radius = _sample_from_cdf(allowed_radii, radii_cdf)
    position = np.random.random(dimensions) * (bounds - 2.0 * radius) + radius

    position_index = np.floor(position / cell_size).astype(np.int64)
    background_grid[
        position_index[0], position_index[1], position_index[2], :dimensions
    ] = position
    background_grid[
        position_index[0], position_index[1], position_index[2], dimensions
    ] = radius

    total_cells = cell_bounds[0] * cell_bounds[1] * cell_bounds[2]
    active_points = np.empty((total_cells, dimensions), dtype=np.float64)
    active_radii = np.empty(total_cells, dtype=np.float64)
    active_count = 1
    active_points[0] = position
    active_radii[0] = radius

    while active_count > 0:
        active_index = np.random.randint(active_count)
        active_position = active_points[active_index]
        active_radius = active_radii[active_index]

        found = False
        for _ in range(repetitions):
            test_radius = _sample_from_cdf(allowed_radii, radii_cdf)
            offset = _random_unit_vector(dimensions)

            r_sum = active_radius + test_radius
            offset *= r_sum * (1.0 + np.random.random())
            test_position = active_position + offset

            # bounds check
            oob = False
            for d in range(dimensions):
                if (
                    test_position[d] < test_radius
                    or test_position[d] > bounds[d] - test_radius
                ):
                    oob = True
                    break
            if oob:
                continue

            # neighbourhood scan
            test_grid_radius = int(np.ceil((rmax + test_radius) / cell_size))
            cell_index = np.floor(test_position / cell_size).astype(np.int64)

            x0 = max(0, cell_index[0] - test_grid_radius)
            x1 = min(cell_bounds[0], cell_index[0] + test_grid_radius + 1)
            y0 = max(0, cell_index[1] - test_grid_radius)
            y1 = min(cell_bounds[1], cell_index[1] + test_grid_radius + 1)
            z0 = max(0, cell_index[2] - test_grid_radius)
            z1 = min(cell_bounds[2], cell_index[2] + test_grid_radius + 1)

            collision = False
            for gx in range(x0, x1):
                for gy in range(y0, y1):
                    for gz in range(z0, z1):
                        r_other = background_grid[gx, gy, gz, dimensions]
                        if np.isnan(r_other):
                            continue

                        dist2 = 0.0
                        for d in range(dimensions):
                            delta = test_position[d] - background_grid[gx, gy, gz, d]
                            dist2 += delta * delta

                        min_dist = test_radius + r_other
                        if dist2 < min_dist * min_dist:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    break

            if collision:
                continue

            # accept point
            background_grid[
                cell_index[0], cell_index[1], cell_index[2], :dimensions
            ] = test_position
            background_grid[cell_index[0], cell_index[1], cell_index[2], dimensions] = (
                test_radius
            )

            active_points[active_count] = test_position
            active_radii[active_count] = test_radius
            active_count += 1
            found = True
            break

        if not found:
            last = active_count - 1
            active_points[active_index] = active_points[last]
            active_radii[active_index] = active_radii[last]
            active_count -= 1

    # Extract samples from the grid
    tmp_points = np.empty((total_cells, dimensions), dtype=np.float64)
    tmp_radii = np.empty(total_cells, dtype=np.float64)
    out_count = 0

    for gx in range(cell_bounds[0]):
        for gy in range(cell_bounds[1]):
            for gz in range(cell_bounds[2]):
                r = background_grid[gx, gy, gz, dimensions]
                if np.isnan(r):
                    continue
                tmp_points[out_count] = background_grid[gx, gy, gz, :dimensions]
                tmp_radii[out_count] = r
                out_count += 1

    return tmp_points[:out_count], tmp_radii[:out_count]


def poisson_disc_samples(
    bounds, radii, repetitions=30, *, rmin_rmax=None, weights=None
):
    """Generate Poisson-disc samples in an N-D axis-aligned box.

    This is a variable-radius variant of Bridson's algorithm (algorithm 4.3).

    Parameters
    ----------
    bounds:
        Array of length N giving box sizes for each dimension.
    radii:
        Either a numpy array of allowed radii (discrete set), or a callable that
        returns a radius.

        Note: the Numba-accelerated path currently supports only the numpy array
        form.
    repetitions:
        Number of candidate samples to try per active point.
    rmin_rmax:
        Required when `radii` is callable. Used to size the background grid.

    Returns
    -------
    (points, radii)
        points: (M, N) array of sample locations
        radii: (M,) radii for each point
    """

    bounds_arr = np.asarray(bounds, dtype=np.float64)

    if isinstance(radii, np.ndarray):
        allowed_radii = np.asarray(radii, dtype=np.float64)
        rmin = float(np.min(allowed_radii))
        rmax = float(np.max(allowed_radii))

        dimensions = bounds_arr.shape[0]
        if dimensions == 2:
            radii_cdf = _prepare_radii_cdf(allowed_radii, weights)
            return _poisson_disc_samples_2d_numba(
                bounds_arr, allowed_radii, radii_cdf, int(repetitions), rmin, rmax
            )
        if dimensions == 3:
            radii_cdf = _prepare_radii_cdf(allowed_radii, weights)
            return _poisson_disc_samples_3d_numba(
                bounds_arr, allowed_radii, radii_cdf, int(repetitions), rmin, rmax
            )

        raise ValueError(
            f"Numba-accelerated implementation currently supports only 2D/3D (got {dimensions}D)"
        )

    if callable(radii):
        if rmin_rmax is None:
            raise ValueError(
                "When `radii` is callable you must provide `rmin_rmax=(rmin, rmax)`"
            )
        raise NotImplementedError(
            "Callable radii is not yet supported by the Numba-accelerated path. "
            "Pass a numpy array of allowed radii for now."
        )

    raise TypeError(
        "`radii` must be a numpy.ndarray of possible radii or a callable. "
        f"Received: {type(radii)!r}"
    )
