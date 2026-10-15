"""Periodic box axes and the minimum surface gap."""

from __future__ import annotations

import math

import numpy as np
import pytest
from click.testing import CliRunner
from packing_asserts import periodic_min_gaps

from polydisc import Ball, Box, Packing, RadiusSpec, nearest_neighbour_gaps, sample
from polydisc.cli import cli
from polydisc.regions import region_from_metadata

SPEC = RadiusSpec.lognormal(gmean=1.0, gstd=math.exp(0.3), rmin=1 / 3, rmax=5 / 3)
TOL = 1e-9


def _check_tiling(pack: Packing, gap: float) -> None:
    """The 3x3 replica has no overlap and no contact closer than `gap`."""
    box = pack.region
    gaps = periodic_min_gaps(pack.points, pack.radii, box.bounds, box.periodic)
    assert gaps.min() >= gap - TOL
    # The fast min-image diagnostic agrees with the brute-force tiling.
    np.testing.assert_allclose(
        nearest_neighbour_gaps(pack.points, pack.radii, box), gaps, atol=1e-9
    )
    for axis, periodic in enumerate(box.periodic):
        x = pack.points[:, axis]
        r = pack.radii
        if periodic:
            assert np.all((x >= 0.0) & (x < box.bounds[axis]))
            # Some particle straddles a face, so wrapping was exercised.
            assert np.any((x < r) | (x > box.bounds[axis] - r))
        else:
            assert np.all(x - r >= -TOL)
            assert np.all(x + r <= box.bounds[axis] + TOL)


@pytest.mark.parametrize("gap", [0.0, 0.3])
def test_2d_fully_periodic_tiles(gap: float) -> None:
    pack = sample(Box([40.0, 30.0], periodic=True), [1.0, 2.0], gap=gap, seed=1)
    _check_tiling(pack, gap)


def test_2d_mixed_axes() -> None:
    pack = sample(Box([40.0, 20.0], periodic=(True, False)), SPEC, gap=0.5, seed=2)
    _check_tiling(pack, 0.5)


@pytest.mark.parametrize("gap", [0.0, 1.0])
def test_3d_periodic_xy_hard_z(gap: float) -> None:
    box = Box([12.0, 12.0, 8.0], periodic=(True, True, False))
    pack = sample(box, SPEC, gap=gap, seed=3)
    _check_tiling(pack, gap)


def test_3d_periodic_z_only() -> None:
    box = Box([10.0, 10.0, 12.0], periodic=(False, False, True))
    pack = sample(box, SPEC, gap=0.25, seed=4)
    _check_tiling(pack, 0.25)


def test_gap_in_plain_box_and_ball() -> None:
    for region in ([30.0, 30.0, 10.0], Ball(8.0)):
        pack = sample(region, SPEC, gap=0.4, seed=5)
        assert nearest_neighbour_gaps(pack.points, pack.radii).min() >= 0.4 - TOL
        assert bool(pack.region.contains(pack.points, pack.radii).all())


def test_gap_survives_spacing() -> None:
    pack = sample([40.0, 40.0], 1.0, gap=0.5, spacing=1.0, seed=6)
    assert nearest_neighbour_gaps(pack.points, pack.radii).min() >= 0.5 - TOL


def test_saturation_comparable_to_hard_walls() -> None:
    # Without walls there is no boundary layer to lose, so a periodic
    # box fills at least about as well as a walled one.
    walled = sample([40.0, 40.0, 10.0], SPEC, gap=0.25, seed=7)
    periodic = sample(
        Box([40.0, 40.0, 10.0], periodic=(True, True, False)), SPEC, gap=0.25, seed=7
    )
    assert periodic.saturated and walled.saturated
    assert periodic.fill_fraction >= 0.97 * walled.fill_fraction
    # A gap lowers the fill, it does not stop the packing from filling.
    loose = sample(
        Box([40.0, 40.0, 10.0], periodic=(True, True, False)), SPEC, gap=1.0, seed=7
    )
    assert 0.3 * periodic.fill_fraction < loose.fill_fraction < periodic.fill_fraction


@pytest.mark.parametrize(
    ("make", "n", "point_sum", "radius_sum"),
    [
        (
            lambda: sample([80.0, 60.0], [1.0, 2.0, 3.0], seed=3),
            402,
            27937.99735707673,
            525.0,
        ),
        (
            lambda: sample(
                [30.0, 30.0, 20.0],
                RadiusSpec.lognormal(gmean=1.0, gstd=1.3, rmax=3.0),
                seed=5,
            ),
            1780,
            71193.6498899043,
            1521.3629558005914,
        ),
        (
            lambda: sample([40.0, 40.0, 4.2], 1.0, seed=1, spacing=1.0),
            107,
            4518.9210909637495,
            107.0,
        ),
        (
            lambda: sample(
                Ball(25.0, dimensions=2), RadiusSpec.uniform(1.0, 2.0), seed=2
            ),
            154,
            7645.685677239943,
            217.00846823164733,
        ),
    ],
)
def test_defaults_unchanged(make, n, point_sum, radius_sum) -> None:
    """Seeded output matches what 0.3.0 produced, before gap/periodic."""
    pack = make()
    assert pack.n == n
    assert float(pack.points.sum()) == pytest.approx(point_sum, rel=1e-12)
    assert float(pack.radii.sum()) == pytest.approx(radius_sum, rel=1e-12)


def test_validation() -> None:
    with pytest.raises(ValueError, match="one flag per axis"):
        Box([10.0, 10.0, 10.0], periodic=(True, False))
    with pytest.raises(ValueError, match="periodic edge x"):
        sample(Box([7.0, 30.0], periodic=(True, False)), 1.0, gap=2.0)
    with pytest.raises(ValueError, match="periodic edge y"):
        sample(Box([30.0, 4.5], periodic=True), 1.0, gap=0.5)
    with pytest.raises(ValueError, match="gap"):
        sample([10.0, 10.0], 1.0, gap=-1.0)
    with pytest.raises(ValueError, match="does not fit"):
        sample(Box([3.5, 30.0], periodic=(True, False)), 1.0)
    with pytest.raises(TypeError):
        Ball(5.0, periodic=True)  # type: ignore[call-arg]


def test_metadata_round_trip(tmp_path) -> None:
    box = Box([20.0, 20.0, 9.0], periodic=(True, True, False))
    assert region_from_metadata(box.describe()) == box
    assert box != Box([20.0, 20.0, 9.0])
    assert "periodic" not in Box([20.0, 20.0]).describe()
    pack = sample(box, 1.0, gap=0.5, seed=0)
    back = Packing.load(pack.save(tmp_path / "p.npz"))
    assert back.region == box
    assert back.gap == 0.5


def test_cli_periodic_gap() -> None:
    runner = CliRunner()
    args = ["generate", "--bounds", "20,20,6", "--periodic", "xy", "--gap", "0.5"]
    result = runner.invoke(cli, [*args, "--radii", "1", "--seed", "0", "--no-report"])
    assert result.exit_code == 0, result.output
    rows = np.loadtxt(result.output.splitlines()[1:], delimiter=",", ndmin=2)
    gaps = periodic_min_gaps(
        rows[:, :3], rows[:, 3], np.array([20.0, 20.0, 6.0]), (True, True, False)
    )
    assert gaps.min() >= 0.5 - 1e-6  # CSV keeps 10 significant digits
    bad = runner.invoke(
        cli,
        [
            "generate",
            "--shape",
            "ball",
            "--ball-radius",
            "5",
            "--periodic",
            "x",
            "--radii",
            "1",
        ],
    )
    assert bad.exit_code != 0
    assert "box" in bad.output
