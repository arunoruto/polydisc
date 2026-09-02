"""Time the samplers across a spread of domains and size distributions.

Run with ``uv run python scripts/benchmark.py``.  Pass ``--compare
<path>`` pointing at a directory holding an ``old_poisson.py`` to print
a side-by-side speedup table against a previous implementation.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np

import polydisc
from polydisc.regions import Ball, Box, Region

CASES: Sequence[tuple[str, Region, np.ndarray]] = (
    ("2D box 1000^2   r=5", Box([1000.0, 1000.0]), np.array([5.0])),
    ("2D box 1000^2   r=2..6 (x3)", Box([1000.0, 1000.0]), np.linspace(2, 6, 5)),
    ("2D box 4000^2   r=1..5 (x5)", Box([4000.0, 4000.0]), np.linspace(1, 5, 9)),
    ("2D box 500^2    r=0.25..5 (x20)", Box([500.0, 500.0]), np.linspace(0.25, 5, 20)),
    ("3D box 100^3    r=5", Box([100.0] * 3), np.array([5.0])),
    ("3D box 150^3    r=1..5 (x5)", Box([150.0] * 3), np.linspace(1, 5, 9)),
    ("3D box 60^3     r=0.5..5 (x10)", Box([60.0] * 3), np.linspace(0.5, 5, 10)),
    ("3D carpet 400^2 x 4.2", Box([400.0, 400.0, 4.2]), np.linspace(1.0, 2.0, 5)),
    ("2D ball R=500   r=1..5 (x5)", Ball(500.0, dimensions=2), np.linspace(1, 5, 9)),
    ("3D ball R=75    r=1..5 (x5)", Ball(75.0), np.linspace(1, 5, 9)),
)


def _time(fn: Callable[[], int]) -> tuple[float, int]:
    """Run `fn` once, returning its wall time and reported count."""
    gc.collect()
    start = time.perf_counter()
    count = fn()
    return time.perf_counter() - start, count


def _peak_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        metavar="DIR",
        default=None,
        help="directory containing an old_poisson.py to compare against",
    )
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args(argv)

    old_sample = None
    if args.compare:
        sys.path.insert(0, args.compare)
        from old_poisson import poisson_disc_samples as old_sample

    # Warm both JITs so compilation does not land in a measurement.
    polydisc.sample([40.0, 40.0], 2.0, repetitions=5, seed=0)
    polydisc.sample([40.0, 40.0, 40.0], 2.0, repetitions=5, seed=0)
    if old_sample is not None:
        np.random.seed(0)
        old_sample(np.array([40.0, 40.0]), np.array([2.0]), 5)
        old_sample(np.array([40.0, 40.0, 40.0]), np.array([2.0]), 5)

    if old_sample is None:
        print(f"{'case':<34}{'N':>9}{'seconds':>10}")
        print("-" * 53)
    else:
        header = f"{'case':<34}{'N':>9}{'old s':>9}{'new s':>9}{'speedup':>9}"
        print(header)
        print("-" * len(header))

    for name, region, radii in CASES:
        new_time, count = _time(
            lambda region=region, radii=radii: (
                polydisc.sample(region, radii, repetitions=args.repetitions, seed=0).n
            )
        )
        if old_sample is None:
            print(f"{name:<34}{count:>9}{new_time:>10.3f}")
            continue

        if not isinstance(region, Box):
            # The old implementation only knew about boxes.
            print(f"{name:<34}{count:>9}{'-':>9}{new_time:>9.3f}{'-':>9}")
            continue

        box = region.bounds
        np.random.seed(0)
        old_time, _ = _time(
            lambda box=box, radii=radii: len(
                old_sample(box, radii, args.repetitions)[0]
            )
        )
        print(
            f"{name:<34}{count:>9}{old_time:>9.3f}{new_time:>9.3f}"
            f"{old_time / new_time:>8.1f}x"
        )

    print(f"\npeak RSS: {_peak_mb():.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
