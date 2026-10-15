"""Numba kernels for variable-radius Poisson-disc sampling.

This module is private; use :mod:`polydisc.sampling` instead.

The sampler is a polydisperse variant of Bridson's algorithm (SIGGRAPH
2007, algorithm 4.3).  Two implementation choices dominate its cost and
are worth stating up front:

Cell size
    The acceleration grid uses ``cell = 2 * rmax``, exactly the largest
    possible query radius ``r_test + rmax``.  The neighbour scan is then
    always a 3x3 (2D) or 3x3x3 (3D) block, *independent of the
    ``rmax / rmin`` ratio*.  Sizing cells by ``rmin`` instead — the
    textbook choice, which buys at most one point per cell — makes the
    scan grow as ``(rmax / rmin) ** d`` and the grid memory grow as
    ``(1 / rmin) ** d``, which is what limits a naive implementation on
    wide size distributions and large domains.

Cell contents
    Because cells are larger than the minimum separation they hold
    several points, stored as an intrusive linked list (``head`` per
    cell, ``next`` per point).  Memory is then proportional to the
    number of *points*, not to the number of grid cells.

Gap and periodic axes
    A minimum surface ``gap`` widens every contact distance to
    ``r_i + r_j + gap`` and the cells to ``2 * rmax + gap``, so the scan
    stays 3x3(x3).  A periodic axis wraps candidates that leave one face
    back in through the other, its cells are widened so a whole number
    of them tiles the length, the neighbour scan wraps around the grid,
    and distances along it use the minimum-image convention.  With
    ``gap = 0`` and no periodic axis every expression reduces to the
    original one exactly, so seeded results are bit-identical.

Everything in the hot loop is scalar: no temporary arrays are allocated
per candidate, which is where a natural NumPy-style translation spends
most of its time under Numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

TWO_PI = 6.283185307179586

#: Directions tried per candidate before giving up on landing inside the
#: container.  A candidate that falls outside is not a failed placement,
#: it is a wasted draw, so charging it against Bridson's ``k`` retires
#: the active point for the wrong reason.  In a slab barely thicker than
#: one particle only a sliver of directions can land inside at all, and
#: without this a monodisperse carpet stops after its very first point.
_BOUNDS_RETRIES = 16

#: Fill an axis-aligned box; see :class:`polydisc.regions.Box`.
REGION_BOX = 0
#: Fill a ball inscribed in the box; see :class:`polydisc.regions.Ball`.
REGION_BALL = 1

#: Sampling ran until the active list emptied (a saturated packing).
STATUS_SATURATED = 0
#: Sampling stopped early because ``max_points`` was reached.
STATUS_MAX_POINTS = 1


@njit(cache=True)
def _draw_radius(kind, values, cdf, p0, p1, rmin, rmax):
    """Draw one radius from the distribution encoded by the arguments.

    Continuous families are truncated to ``[rmin, rmax]`` by rejection.
    The rejection loop is bounded; on the (astronomically unlikely)
    exhaustion path it falls back to a uniform draw inside the window so
    the sampler can never hang.
    """
    if kind == 0:  # discrete
        u = np.random.random()
        lo = 0
        hi = cdf.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        return values[lo]

    if kind == 3:  # uniform
        return rmin + np.random.random() * (rmax - rmin)

    for _ in range(256):
        if kind == 1:  # lognormal
            r = np.exp(p0 + p1 * np.random.normal(0.0, 1.0))
        else:  # normal
            r = p0 + p1 * np.random.normal(0.0, 1.0)
        if rmin <= r <= rmax:
            return r

    return rmin + np.random.random() * (rmax - rmin)


@njit(cache=True)
def _random_direction_3d():
    """Draw a uniform direction on the unit sphere.

    Marsaglia (1972): two uniforms, no trigonometry and no temporary
    array, which matters in a loop run millions of times.
    """
    while True:
        u1 = 2.0 * np.random.random() - 1.0
        u2 = 2.0 * np.random.random() - 1.0
        s = u1 * u1 + u2 * u2
        if s < 1.0:
            break
    f = 2.0 * np.sqrt(1.0 - s)
    return u1 * f, u2 * f, 1.0 - 2.0 * s


@njit(cache=True)
def _periodic_cells(length, cell):
    """Cell count and widened cell size that tile a periodic `length`."""
    count = max(1, int(length // cell))
    return count, length / count


@njit(cache=True)
def _wrap(x, length):
    """Map `x` into ``[0, length)``."""
    x -= length * np.floor(x / length)
    if x >= length:
        x -= length
    return x


@njit(cache=True)
def _scan_range(c, g, count, periodic):
    """Cell offsets to scan along one axis, ``[lo, hi)``.

    On a periodic axis the offsets may run off the grid and are wrapped
    by the caller; a window that would visit a cell twice covers the
    whole axis instead.
    """
    if periodic:
        if 2 * g + 1 >= count:
            return 0, count
        return c - g, c + g + 1
    return max(0, c - g), min(count, c + g + 1)


@njit(cache=True)
def _wrap_cell(g, count):
    """Wrap a cell offset back onto ``[0, count)``."""
    if g < 0:
        return g + count
    if g >= count:
        return g - count
    return g


@njit(cache=True)
def _min_image(d, length, periodic):
    """Minimum-image component of a separation along one axis."""
    if periodic:
        return d - length * np.floor(d / length + 0.5)
    return d


@njit(cache=True)
def _draw_radii(count, kind, values, cdf, p0, p1, rmin, rmax, seed):
    """Draw `count` independent radii from one distribution.

    Shares :func:`_draw_radius` with the samplers, so a check on this
    output is a check on the radii the samplers actually see.
    """
    np.random.seed(seed)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        out[i] = _draw_radius(kind, values, cdf, p0, p1, rmin, rmax)
    return out


@njit(cache=True)
def _sample_2d(
    bx,
    by,
    region,
    kind,
    values,
    cdf,
    p0,
    p1,
    rmin,
    rmax,
    k,
    seed,
    max_points,
    gap=0.0,
    perx=False,
    pery=False,
):
    """Sample non-overlapping discs inside a 2D region.

    The region is either the box ``[0, bx] x [0, by]`` or the disc
    inscribed in it, selected by `region`; the acceleration grid spans
    the box either way, which just leaves the corner cells empty.

    Every pair ends up at least ``r_i + r_j + gap`` apart; `perx` and
    `pery` make that axis periodic (box only).

    Returns ``(points, radii, status)``; see :data:`STATUS_SATURATED`.
    Callers must guarantee that the largest particle fits, and that a
    periodic length is at least ``2 * (2 * rmax + gap)``.
    """
    np.random.seed(seed)

    cell = 2.0 * rmax + gap
    nx = max(1, int(np.ceil(bx / cell)))
    ny = max(1, int(np.ceil(by / cell)))
    cellx = cell
    celly = cell
    if perx:
        nx, cellx = _periodic_cells(bx, cell)
    if pery:
        ny, celly = _periodic_cells(by, cell)
    head = np.full(nx * ny, -1, dtype=np.int64)

    # Ball centre and radius, unused for a box region.
    hx = 0.5 * bx
    hy = 0.5 * by

    cap = min(max_points, 1024)
    pts = np.empty((cap, 2), dtype=np.float64)
    rad = np.empty(cap, dtype=np.float64)
    nxt = np.empty(cap, dtype=np.int64)

    acap = min(max_points, 1024)
    active = np.empty(acap, dtype=np.int64)

    # Seed point: uniform over the region where a disc of this radius
    # fits.  For a ball that shrunken region is itself a disc, sampled
    # directly rather than by rejection -- rejection would stall when
    # the particle nearly fills the container.
    r0 = _draw_radius(kind, values, cdf, p0, p1, rmin, rmax)
    if region == REGION_BALL:
        theta = TWO_PI * np.random.random()
        span = (hx - r0) * np.sqrt(np.random.random())
        x0 = hx + span * np.cos(theta)
        y0 = hy + span * np.sin(theta)
    else:
        if perx:
            x0 = np.random.random() * bx
        else:
            x0 = r0 + np.random.random() * (bx - 2.0 * r0)
        if pery:
            y0 = np.random.random() * by
        else:
            y0 = r0 + np.random.random() * (by - 2.0 * r0)

    cx = min(nx - 1, max(0, int(x0 / cellx)))
    cy = min(ny - 1, max(0, int(y0 / celly)))
    pts[0, 0] = x0
    pts[0, 1] = y0
    rad[0] = r0
    nxt[0] = -1
    head[cx * ny + cy] = 0
    n = 1
    active[0] = 0
    na = 1

    status = STATUS_SATURATED

    while na > 0:
        if n >= max_points:
            status = STATUS_MAX_POINTS
            break

        ai = np.random.randint(na)
        pi = active[ai]
        px = pts[pi, 0]
        py = pts[pi, 1]
        pr = rad[pi]

        found = False
        for _ in range(k):
            tr = _draw_radius(kind, values, cdf, p0, p1, rmin, rmax)

            # Candidate in the annulus [r_sum, 2 * r_sum] around the
            # active point, with r_sum = pr + tr the contact distance.
            # Redrawing the direction until it lands inside samples the
            # same annulus-intersect-container region as rejecting would;
            # it just does not spend a candidate doing so.
            tx = 0.0
            ty = 0.0
            outside = True
            for _ in range(_BOUNDS_RETRIES):
                theta = TWO_PI * np.random.random()
                dist = (pr + tr + gap) * (1.0 + np.random.random())
                tx = px + dist * np.cos(theta)
                ty = py + dist * np.sin(theta)

                if region == REGION_BALL:
                    ddx = tx - hx
                    ddy = ty - hy
                    span = hx - tr
                    outside = ddx * ddx + ddy * ddy > span * span
                else:
                    # A periodic axis wraps the candidate (snake-style)
                    # instead of rejecting it.
                    if perx:
                        tx = _wrap(tx, bx)
                    if pery:
                        ty = _wrap(ty, by)
                    outside = (not perx and (tx < tr or tx > bx - tr)) or (
                        not pery and (ty < tr or ty > by - tr)
                    )
                if not outside:
                    break
            if outside:
                continue

            cx = min(nx - 1, max(0, int(tx / cellx)))
            cy = min(ny - 1, max(0, int(ty / celly)))
            reach = tr + rmax + gap
            gx0, gx1 = _scan_range(cx, int(np.ceil(reach / cellx)), nx, perx)
            gy0, gy1 = _scan_range(cy, int(np.ceil(reach / celly)), ny, pery)

            collision = False
            for ox in range(gx0, gx1):
                base = _wrap_cell(ox, nx) * ny
                for oy in range(gy0, gy1):
                    j = head[base + _wrap_cell(oy, ny)]
                    while j != -1:
                        dx = _min_image(tx - pts[j, 0], bx, perx)
                        dy = _min_image(ty - pts[j, 1], by, pery)
                        md = tr + rad[j] + gap
                        if dx * dx + dy * dy < md * md:
                            collision = True
                            break
                        j = nxt[j]
                    if collision:
                        break
                if collision:
                    break
            if collision:
                continue

            if n == cap:
                newcap = cap * 2
                newcap = min(newcap, max_points)
                new_pts = np.empty((newcap, 2), dtype=np.float64)
                new_rad = np.empty(newcap, dtype=np.float64)
                new_nxt = np.empty(newcap, dtype=np.int64)
                new_pts[:n] = pts[:n]
                new_rad[:n] = rad[:n]
                new_nxt[:n] = nxt[:n]
                pts = new_pts
                rad = new_rad
                nxt = new_nxt
                cap = newcap

            pts[n, 0] = tx
            pts[n, 1] = ty
            rad[n] = tr
            cidx = cx * ny + cy
            nxt[n] = head[cidx]
            head[cidx] = n

            if na == acap:
                newacap = acap * 2
                newacap = min(newacap, max_points)
                new_active = np.empty(newacap, dtype=np.int64)
                new_active[:na] = active[:na]
                active = new_active
                acap = newacap

            active[na] = n
            na += 1
            n += 1
            found = True
            break

        if not found:
            na -= 1
            active[ai] = active[na]

    return pts[:n].copy(), rad[:n].copy(), status


@njit(cache=True)
def _sample_3d(
    bx,
    by,
    bz,
    region,
    kind,
    values,
    cdf,
    p0,
    p1,
    rmin,
    rmax,
    k,
    seed,
    max_points,
    gap=0.0,
    perx=False,
    pery=False,
    perz=False,
):
    """Sample non-overlapping spheres inside a 3D region.

    The region is either the box ``[0, bx] x [0, by] x [0, bz]`` or the
    ball inscribed in it, selected by `region`.

    `gap`, `perx`, `pery` and `perz` work as in :func:`_sample_2d`.

    Returns ``(points, radii, status)``.  Callers must guarantee that
    the largest particle fits.
    """
    np.random.seed(seed)

    cell = 2.0 * rmax + gap
    nx = max(1, int(np.ceil(bx / cell)))
    ny = max(1, int(np.ceil(by / cell)))
    nz = max(1, int(np.ceil(bz / cell)))
    cellx = cell
    celly = cell
    cellz = cell
    if perx:
        nx, cellx = _periodic_cells(bx, cell)
    if pery:
        ny, celly = _periodic_cells(by, cell)
    if perz:
        nz, cellz = _periodic_cells(bz, cell)
    head = np.full(nx * ny * nz, -1, dtype=np.int64)

    # Ball centre and radius, unused for a box region.
    hx = 0.5 * bx
    hy = 0.5 * by
    hz = 0.5 * bz

    cap = min(max_points, 1024)
    pts = np.empty((cap, 3), dtype=np.float64)
    rad = np.empty(cap, dtype=np.float64)
    nxt = np.empty(cap, dtype=np.int64)

    acap = min(max_points, 1024)
    active = np.empty(acap, dtype=np.int64)

    r0 = _draw_radius(kind, values, cdf, p0, p1, rmin, rmax)
    if region == REGION_BALL:
        ux, uy, uz = _random_direction_3d()
        span = (hx - r0) * np.cbrt(np.random.random())
        x0 = hx + span * ux
        y0 = hy + span * uy
        z0 = hz + span * uz
    else:
        if perx:
            x0 = np.random.random() * bx
        else:
            x0 = r0 + np.random.random() * (bx - 2.0 * r0)
        if pery:
            y0 = np.random.random() * by
        else:
            y0 = r0 + np.random.random() * (by - 2.0 * r0)
        if perz:
            z0 = np.random.random() * bz
        else:
            z0 = r0 + np.random.random() * (bz - 2.0 * r0)

    cx = min(nx - 1, max(0, int(x0 / cellx)))
    cy = min(ny - 1, max(0, int(y0 / celly)))
    cz = min(nz - 1, max(0, int(z0 / cellz)))
    pts[0, 0] = x0
    pts[0, 1] = y0
    pts[0, 2] = z0
    rad[0] = r0
    nxt[0] = -1
    head[(cx * ny + cy) * nz + cz] = 0
    n = 1
    active[0] = 0
    na = 1

    status = STATUS_SATURATED

    while na > 0:
        if n >= max_points:
            status = STATUS_MAX_POINTS
            break

        ai = np.random.randint(na)
        pi = active[ai]
        px = pts[pi, 0]
        py = pts[pi, 1]
        pz = pts[pi, 2]
        pr = rad[pi]

        found = False
        for _ in range(k):
            tr = _draw_radius(kind, values, cdf, p0, p1, rmin, rmax)

            tx = 0.0
            ty = 0.0
            tz = 0.0
            outside = True
            for _ in range(_BOUNDS_RETRIES):
                dist = (pr + tr + gap) * (1.0 + np.random.random())

                if region == REGION_BALL:
                    ux, uy, uz = _random_direction_3d()
                    tx = px + dist * ux
                    ty = py + dist * uy
                    tz = pz + dist * uz
                    ddx = tx - hx
                    ddy = ty - hy
                    ddz = tz - hz
                    span = hx - tr
                    outside = ddx * ddx + ddy * ddy + ddz * ddz > span * span
                else:
                    # Archimedes' hat-box theorem: for a direction drawn
                    # uniformly on the sphere, the z coordinate is
                    # uniform along the axis.  So drawing z directly
                    # inside the slab, with the azimuth still uniform,
                    # samples *exactly* the in-slab directions -- no
                    # rejection at all.  That is what lets a carpet fill:
                    # in a box barely thicker than one particle, a blind
                    # direction lands inside perhaps one time in eighty.
                    # A periodic z needs no clipping, only wrapping.
                    if perz:
                        zlo = pz - dist
                        zhi = pz + dist
                    else:
                        zlo = max(pz - dist, tr)
                        zhi = min(pz + dist, bz - tr)
                    if zhi < zlo:
                        continue
                    tz = zlo + np.random.random() * (zhi - zlo)
                    uz = (tz - pz) / dist
                    horizontal = dist * np.sqrt(max(1.0 - uz * uz, 0.0))
                    theta = TWO_PI * np.random.random()
                    tx = px + horizontal * np.cos(theta)
                    ty = py + horizontal * np.sin(theta)
                    if perx:
                        tx = _wrap(tx, bx)
                    if pery:
                        ty = _wrap(ty, by)
                    if perz:
                        tz = _wrap(tz, bz)
                    outside = (not perx and (tx < tr or tx > bx - tr)) or (
                        not pery and (ty < tr or ty > by - tr)
                    )
                if not outside:
                    break
            if outside:
                continue

            cx = min(nx - 1, max(0, int(tx / cellx)))
            cy = min(ny - 1, max(0, int(ty / celly)))
            cz = min(nz - 1, max(0, int(tz / cellz)))
            reach = tr + rmax + gap
            gx0, gx1 = _scan_range(cx, int(np.ceil(reach / cellx)), nx, perx)
            gy0, gy1 = _scan_range(cy, int(np.ceil(reach / celly)), ny, pery)
            gz0, gz1 = _scan_range(cz, int(np.ceil(reach / cellz)), nz, perz)

            collision = False
            for ox in range(gx0, gx1):
                gx = _wrap_cell(ox, nx)
                for oy in range(gy0, gy1):
                    base = (gx * ny + _wrap_cell(oy, ny)) * nz
                    for oz in range(gz0, gz1):
                        j = head[base + _wrap_cell(oz, nz)]
                        while j != -1:
                            dx = _min_image(tx - pts[j, 0], bx, perx)
                            dy = _min_image(ty - pts[j, 1], by, pery)
                            dz = _min_image(tz - pts[j, 2], bz, perz)
                            md = tr + rad[j] + gap
                            if dx * dx + dy * dy + dz * dz < md * md:
                                collision = True
                                break
                            j = nxt[j]
                        if collision:
                            break
                    if collision:
                        break
                if collision:
                    break
            if collision:
                continue

            if n == cap:
                newcap = cap * 2
                newcap = min(newcap, max_points)
                new_pts = np.empty((newcap, 3), dtype=np.float64)
                new_rad = np.empty(newcap, dtype=np.float64)
                new_nxt = np.empty(newcap, dtype=np.int64)
                new_pts[:n] = pts[:n]
                new_rad[:n] = rad[:n]
                new_nxt[:n] = nxt[:n]
                pts = new_pts
                rad = new_rad
                nxt = new_nxt
                cap = newcap

            pts[n, 0] = tx
            pts[n, 1] = ty
            pts[n, 2] = tz
            rad[n] = tr
            cidx = (cx * ny + cy) * nz + cz
            nxt[n] = head[cidx]
            head[cidx] = n

            if na == acap:
                newacap = acap * 2
                newacap = min(newacap, max_points)
                new_active = np.empty(newacap, dtype=np.int64)
                new_active[:na] = active[:na]
                active = new_active
                acap = newacap

            active[na] = n
            na += 1
            n += 1
            found = True
            break

        if not found:
            na -= 1
            active[ai] = active[na]

    return pts[:n].copy(), rad[:n].copy(), status


@njit(cache=True)
def _build_grid(pts, cell, nx, ny, nz):
    """Bin points into a linked-list cell grid.

    Parameters
    ----------
    pts : (N, 3) array
        Point coordinates, already shifted to a non-negative box.
    cell : float
        Cell edge length.
    nx, ny, nz : int
        Cell counts per axis.

    Returns
    -------
    (head, nxt) : tuple of numpy.ndarray
        ``head`` holds the first point index per cell (``-1`` if empty)
        and ``nxt`` chains the remaining points of that cell.
    """
    head = np.full(nx * ny * nz, -1, dtype=np.int64)
    nxt = np.empty(pts.shape[0], dtype=np.int64)
    for i in range(pts.shape[0]):
        cx = min(nx - 1, max(0, int(pts[i, 0] / cell)))
        cy = min(ny - 1, max(0, int(pts[i, 1] / cell)))
        cz = min(nz - 1, max(0, int(pts[i, 2] / cell)))
        c = (cx * ny + cy) * nz + cz
        nxt[i] = head[c]
        head[c] = i
    return head, nxt


@njit(cache=True, parallel=True)
def _nearest_gaps(pts, rad, cell, nx, ny, nz, head, nxt, rmax):
    """Surface-to-surface gap to each point's nearest neighbour.

    For every point ``i`` this returns
    ``min_{j != i} (|x_i - x_j| - r_i - r_j)``, negative where spheres
    overlap.  Cells are visited in Chebyshev shells of growing radius
    and the search stops as soon as no unvisited shell could improve on
    the best gap found so far, so the cost stays local even though the
    answer is exact.

    Parameters
    ----------
    pts : (N, 3) array
        Point coordinates, shifted so all are non-negative.
    rad : (N,) array
        Radii.
    cell : float
        Grid cell edge length.
    nx, ny, nz : int
        Cell counts per axis.
    head, nxt : numpy.ndarray
        Linked-list grid from :func:`_build_grid`.
    rmax : float
        Upper bound on any radius, used for the shell cut-off.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` array of gaps; ``inf`` for a single isolated point.
    """
    n = pts.shape[0]
    out = np.empty(n, dtype=np.float64)
    max_shell = max(nx, max(ny, nz))

    for i in prange(n):
        xi = pts[i, 0]
        yi = pts[i, 1]
        zi = pts[i, 2]
        ri = rad[i]
        cx = min(nx - 1, max(0, int(xi / cell)))
        cy = min(ny - 1, max(0, int(yi / cell)))
        cz = min(nz - 1, max(0, int(zi / cell)))

        best = np.inf
        for shell in range(max_shell + 1):
            # Anything in an unvisited shell is at least this far away.
            if shell > 0 and best <= (shell - 1) * cell - ri - rmax:
                break

            gx0 = max(0, cx - shell)
            gx1 = min(nx - 1, cx + shell)
            gy0 = max(0, cy - shell)
            gy1 = min(ny - 1, cy + shell)
            gz0 = max(0, cz - shell)
            gz1 = min(nz - 1, cz + shell)

            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    for gz in range(gz0, gz1 + 1):
                        # Only the surface of the cube: inner cells were
                        # covered by an earlier shell.
                        if shell > 0:
                            on_shell = (
                                gx == cx - shell
                                or gx == cx + shell
                                or gy == cy - shell
                                or gy == cy + shell
                                or gz == cz - shell
                                or gz == cz + shell
                            )
                            if not on_shell:
                                continue
                        j = head[(gx * ny + gy) * nz + gz]
                        while j != -1:
                            if j != i:
                                dx = xi - pts[j, 0]
                                dy = yi - pts[j, 1]
                                dz = zi - pts[j, 2]
                                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                                gap = d - ri - rad[j]
                                best = min(best, gap)
                            j = nxt[j]
        out[i] = best
    return out
