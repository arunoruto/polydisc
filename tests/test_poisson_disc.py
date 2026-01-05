import numpy as np
import pytest

from pysson_disc.poisson_disc import poisson_disc_samples


def test_cli_generate_config_from_stdin_toml() -> None:
    from click.testing import CliRunner

    from pysson_disc.cli import cli

    runner = CliRunner()

    config_toml = """\
dims = 2
bounds = [50.0, 50.0]
radii = [5.0, 10.0]
repetitions = 5
seed = 0
"""

    result = runner.invoke(
        cli,
        ["generate", "--config", "-", "--out", "-"],
        input=config_toml,
    )
    assert result.exit_code == 0, result.output

    # CSV data output (not config)
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert lines[0].strip() == "x,y,r"
    assert len(lines) >= 2


def test_parse_gaussian_string() -> None:
    from pysson_disc.cli import _parse_gaussian

    mean, sigma = _parse_gaussian("10,2", option_name="--gaussian")
    assert mean == 10.0
    assert sigma == 2.0


def test_parse_gaussian_object() -> None:
    from pysson_disc.cli import _parse_gaussian

    mean, sigma = _parse_gaussian({"mean": 10, "sigma": 2}, option_name="--gaussian")
    assert mean == 10.0
    assert sigma == 2.0


def _assert_in_bounds(
    points: np.ndarray, radii: np.ndarray, bounds: np.ndarray
) -> None:
    assert points.shape[0] == radii.shape[0]
    assert points.shape[1] == bounds.shape[0]
    assert np.all(points >= radii[:, None])
    assert np.all(points <= bounds[None, :] - radii[:, None])


def _assert_no_overlaps(points: np.ndarray, radii: np.ndarray) -> None:
    if len(points) <= 1:
        return

    diffs = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=-1)
    min_dist2 = (radii[:, None] + radii[None, :]) ** 2

    mask = ~np.eye(len(points), dtype=bool)
    assert np.all(dist2[mask] >= min_dist2[mask])


def test_samples_respect_bounds_and_distance_with_array_radii_2d() -> None:
    np.random.seed(0)
    bounds = np.array([200.0, 100.0])
    allowed_radii = np.array([5.0, 10.0, 15.0])

    points, radii = poisson_disc_samples(bounds, allowed_radii, repetitions=10)

    assert len(points) == len(radii)
    assert points.shape[1] == 2

    _assert_in_bounds(points, radii, bounds)
    _assert_no_overlaps(points, radii)


def test_samples_respect_bounds_and_distance_with_array_radii_3d() -> None:
    np.random.seed(0)
    bounds = np.array([60.0, 60.0, 60.0])
    allowed_radii = np.array([3.0, 4.0])

    points, radii = poisson_disc_samples(bounds, allowed_radii, repetitions=10)

    assert len(points) == len(radii)
    assert points.shape[1] == 3

    _assert_in_bounds(points, radii, bounds)
    _assert_no_overlaps(points, radii)


def test_callable_radii_requires_range() -> None:
    np.random.seed(0)
    bounds = np.array([100.0, 100.0])

    def radii_fn() -> float:
        return float(np.random.uniform(5.0, 10.0))

    with pytest.raises(ValueError, match="rmin_rmax"):
        poisson_disc_samples(bounds, radii_fn, repetitions=5)


def test_callable_radii_with_range_not_implemented_yet() -> None:
    np.random.seed(0)
    bounds = np.array([120.0, 120.0])

    def radii_fn() -> float:
        return float(np.random.uniform(5.0, 10.0))

    with pytest.raises(NotImplementedError):
        poisson_disc_samples(bounds, radii_fn, repetitions=10, rmin_rmax=(5.0, 10.0))
