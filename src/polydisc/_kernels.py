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

Everything in the hot loop is scalar: no temporary arrays are allocated
per candidate, which is where a natural NumPy-style translation spends
most of its time under Numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

TWO_PI = 6.283185307179586

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
    bx, by, region, kind, values, cdf, p0, p1, rmin, rmax, k, seed, max_points
):
    """Sample non-overlapping discs inside a 2D region.

    The region is either the box ``[0, bx] x [0, by]`` or the disc
    inscribed in it, selected by `region`; the acceleration grid spans
    the box either way, which just leaves the corner cells empty.

    Returns ``(points, radii, status)``; see :data:`STATUS_SATURATED`.
    Callers must guarantee that the largest particle fits.
    """
    np.random.seed(seed)

    cell = 2.0 * rmax
    nx = max(1, int(np.ceil(bx / cell)))
    ny = max(1, int(np.ceil(by / cell)))
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
        x0 = r0 + np.random.random() * (bx - 2.0 * r0)
        y0 = r0 + np.random.random() * (by - 2.0 * r0)

    cx = min(nx - 1, max(0, int(x0 / cell)))
    cy = min(ny - 1, max(0, int(y0 / cell)))
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
            theta = TWO_PI * np.random.random()
            dist = (pr + tr) * (1.0 + np.random.random())
            tx = px + dist * np.cos(theta)
            ty = py + dist * np.sin(theta)

            if region == REGION_BALL:
                ddx = tx - hx
                ddy = ty - hy
                span = hx - tr
                outside = ddx * ddx + ddy * ddy > span * span
            else:
                outside = tx < tr or tx > bx - tr or ty < tr or ty > by - tr
            if outside:
                continue

            cx = min(nx - 1, max(0, int(tx / cell)))
            cy = min(ny - 1, max(0, int(ty / cell)))
            gr = int(np.ceil((tr + rmax) / cell))

            gx0 = max(0, cx - gr)
            gx1 = min(nx, cx + gr + 1)
            gy0 = max(0, cy - gr)
            gy1 = min(ny, cy + gr + 1)

            collision = False
            for gx in range(gx0, gx1):
                base = gx * ny
                for gy in range(gy0, gy1):
                    j = head[base + gy]
                    while j != -1:
                        dx = tx - pts[j, 0]
                        dy = ty - pts[j, 1]
                        md = tr + rad[j]
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
    bx, by, bz, region, kind, values, cdf, p0, p1, rmin, rmax, k, seed, max_points
):
    """Sample non-overlapping spheres inside a 3D region.

    The region is either the box ``[0, bx] x [0, by] x [0, bz]`` or the
    ball inscribed in it, selected by `region`.

    Returns ``(points, radii, status)``.  Callers must guarantee that
    the largest particle fits.
    """
    np.random.seed(seed)

    cell = 2.0 * rmax
    nx = max(1, int(np.ceil(bx / cell)))
    ny = max(1, int(np.ceil(by / cell)))
    nz = max(1, int(np.ceil(bz / cell)))
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
        x0 = r0 + np.random.random() * (bx - 2.0 * r0)
        y0 = r0 + np.random.random() * (by - 2.0 * r0)
        z0 = r0 + np.random.random() * (bz - 2.0 * r0)

    cx = min(nx - 1, max(0, int(x0 / cell)))
    cy = min(ny - 1, max(0, int(y0 / cell)))
    cz = min(nz - 1, max(0, int(z0 / cell)))
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

            ux, uy, uz = _random_direction_3d()

            dist = (pr + tr) * (1.0 + np.random.random())
            tx = px + dist * ux
            ty = py + dist * uy
            tz = pz + dist * uz

            if region == REGION_BALL:
                ddx = tx - hx
                ddy = ty - hy
                ddz = tz - hz
                span = hx - tr
                outside = ddx * ddx + ddy * ddy + ddz * ddz > span * span
            else:
                outside = (
                    tx < tr
                    or tx > bx - tr
                    or ty < tr
                    or ty > by - tr
                    or tz < tr
                    or tz > bz - tr
                )
            if outside:
                continue

            cx = min(nx - 1, max(0, int(tx / cell)))
            cy = min(ny - 1, max(0, int(ty / cell)))
            cz = min(nz - 1, max(0, int(tz / cell)))
            gr = int(np.ceil((tr + rmax) / cell))

            gx0 = max(0, cx - gr)
            gx1 = min(nx, cx + gr + 1)
            gy0 = max(0, cy - gr)
            gy1 = min(ny, cy + gr + 1)
            gz0 = max(0, cz - gr)
            gz1 = min(nz, cz + gr + 1)

            collision = False
            for gx in range(gx0, gx1):
                for gy in range(gy0, gy1):
                    base = (gx * ny + gy) * nz
                    for gz in range(gz0, gz1):
                        j = head[base + gz]
                        while j != -1:
                            dx = tx - pts[j, 0]
                            dy = ty - pts[j, 1]
                            dz = tz - pts[j, 2]
                            md = tr + rad[j]
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
