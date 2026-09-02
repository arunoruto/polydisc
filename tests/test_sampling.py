"""Tests for :mod:`polydisc.sampling`."""

from __future__ import annotations

import numpy as np
import pytest
from packing_asserts import assert_inside, assert_no_overlaps

from polydisc.radii import RadiusSpec
from polydisc.sampling import (
    Packing,
    geometric_bias,
    poisson_disc_samples,
    sample,
)


class TestGeometry:
    @pytest.mark.parametrize(
        ("bounds", "radii"),
        [
            ([120.0, 80.0], 5.0),
            ([120.0, 80.0], [3.0, 5.0, 8.0]),
            ([50.0, 50.0, 50.0], 4.0),
            ([50.0, 50.0, 50.0], [2.0, 5.0]),
            ([80.0, 60.0], RadiusSpec.lognormal(3.0, 1.4)),
            ([40.0, 40.0, 40.0], RadiusSpec.uniform(1.5, 4.0)),
        ],
        ids=["2d-mono", "2d-poly", "3d-mono", "3d-poly", "2d-lognormal", "3d-uniform"],
    )
    def test_packing_is_valid(self, bounds, radii) -> None:
        pack = sample(bounds, radii, repetitions=15, seed=0)
        assert pack.n > 0
        assert_inside(pack.points, pack.radii, np.asarray(bounds, dtype=float))
        assert_no_overlaps(pack.points, pack.radii)

    def test_radii_stay_inside_the_spec_window(self) -> None:
        spec = RadiusSpec.lognormal(2.0, 1.5)
        pack = sample([80.0, 80.0], spec, seed=1)
        assert pack.radii.min() >= spec.rmin
        assert pack.radii.max() <= spec.rmax

    def test_weights_steer_the_realised_radii(self) -> None:
        heavy_small = sample([120.0, 120.0], [2.0, 6.0], weights=[9.0, 1.0], seed=2)
        heavy_large = sample([120.0, 120.0], [2.0, 6.0], weights=[1.0, 9.0], seed=2)
        assert heavy_small.radii.mean() < heavy_large.radii.mean()

    def test_saturated_packing_is_dense(self) -> None:
        pack = sample([200.0, 200.0], 4.0, repetitions=30, seed=3)
        assert pack.saturated
        # Saturated 2D Poisson-disc packings sit well above half the
        # domain; anything much lower means placement is failing early.
        assert pack.fill_fraction > 0.4


class TestDeterminism:
    def test_same_seed_reproduces_the_packing(self) -> None:
        first = sample([80.0, 80.0], [2.0, 4.0], seed=99)
        second = sample([80.0, 80.0], [2.0, 4.0], seed=99)
        np.testing.assert_array_equal(first.points, second.points)
        np.testing.assert_array_equal(first.radii, second.radii)

    def test_different_seeds_differ(self) -> None:
        first = sample([80.0, 80.0], [2.0, 4.0], seed=1)
        second = sample([80.0, 80.0], [2.0, 4.0], seed=2)
        assert first.points.shape != second.points.shape or not np.array_equal(
            first.points, second.points
        )

    def test_omitted_seed_is_recorded_and_replayable(self) -> None:
        # The seed must be an output, not just an input: this is what
        # makes an exploratory run reproducible after the fact.
        pack = sample([60.0, 60.0], 3.0)
        replay = sample([60.0, 60.0], 3.0, seed=pack.seed)
        np.testing.assert_array_equal(pack.points, replay.points)

    def test_numpy_global_seed_drives_the_default_seed(self) -> None:
        np.random.seed(1234)
        first = sample([60.0, 60.0], 3.0)
        np.random.seed(1234)
        second = sample([60.0, 60.0], 3.0)
        np.testing.assert_array_equal(first.points, second.points)


class TestLimits:
    def test_max_points_stops_early(self) -> None:
        pack = sample([200.0, 200.0], 3.0, seed=0, max_points=40)
        assert pack.n <= 40
        assert not pack.saturated

    def test_max_points_beyond_saturation_is_a_no_op(self) -> None:
        capped = sample([60.0, 60.0], 4.0, seed=0, max_points=10_000)
        uncapped = sample([60.0, 60.0], 4.0, seed=0)
        assert capped.saturated
        np.testing.assert_array_equal(capped.points, uncapped.points)

    def test_a_box_barely_larger_than_one_particle_still_works(self) -> None:
        pack = sample([2.5, 2.5], 1.0, seed=0)
        assert pack.n == 1
        assert_inside(pack.points, pack.radii, np.array([2.5, 2.5]))


class TestValidation:
    @pytest.mark.parametrize(
        ("bounds", "match"),
        [
            ([10.0], "2 or 3 entries"),
            ([10.0, 10.0, 10.0, 10.0], "2 or 3 entries"),
            ([10.0, 0.0], "must all be > 0"),
            ([10.0, -1.0], "must all be > 0"),
            ([10.0, np.nan], "must all be finite"),
        ],
    )
    def test_bad_bounds_are_rejected(self, bounds, match) -> None:
        with pytest.raises(ValueError, match=match):
            sample(bounds, 1.0)

    def test_particle_larger_than_the_box_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            sample([10.0, 10.0], 5.0)

    def test_repetitions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="repetitions must be >= 1"):
            sample([50.0, 50.0], 2.0, repetitions=0)

    def test_max_points_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_points must be >= 1"):
            sample([50.0, 50.0], 2.0, max_points=0)

    def test_spacing_shrink_failure_names_the_cause(self) -> None:
        # 8 wide, radius 2: fits at spacing 0, but not once spacing
        # pre-shrinks the box, and the message has to say so.
        with pytest.raises(ValueError, match="fit_bounds=False"):
            sample([8.0, 8.0], 2.0, spacing=2.0)


class TestPacking:
    def test_fill_fraction_matches_a_hand_computation(self) -> None:
        pack = sample([100.0, 100.0], 5.0, seed=0)
        expected = float(np.sum(np.pi * pack.radii**2) / (100.0 * 100.0))
        assert pack.fill_fraction == pytest.approx(expected)

    def test_metadata_is_json_ready(self) -> None:
        import json

        pack = sample([60.0, 60.0], [2.0, 3.0], seed=5, spacing=1.0)
        restored = json.loads(json.dumps(pack.metadata()))
        assert restored["n"] == pack.n
        assert restored["spacing"] == 1.0
        assert restored["radii"]["kind"] == "discrete"

    def test_save_load_roundtrip(self, tmp_path) -> None:
        pack = sample([60.0, 60.0, 20.0], RadiusSpec.lognormal(2.0, 1.3), seed=6)
        path = pack.save(tmp_path / "nested" / "pack.npz")
        loaded = Packing.load(path)

        np.testing.assert_array_equal(loaded.points, pack.points)
        np.testing.assert_array_equal(loaded.radii, pack.radii)
        np.testing.assert_array_equal(loaded.bounds, pack.bounds)
        assert loaded.seed == pack.seed
        assert loaded.spec.kind == pack.spec.kind
        assert loaded.spec.rmin == pytest.approx(pack.spec.rmin)


class TestLegacyWrapper:
    def test_returns_bare_arrays(self) -> None:
        points, radii = poisson_disc_samples([60.0, 60.0], np.array([3.0]), 10, seed=0)
        assert points.shape[0] == radii.shape[0]
        assert points.shape[1] == 2

    def test_matches_sample(self) -> None:
        points, radii = poisson_disc_samples([60.0, 60.0], 3.0, seed=8)
        pack = sample([60.0, 60.0], 3.0, repetitions=30, seed=8)
        np.testing.assert_array_equal(points, pack.points)
        np.testing.assert_array_equal(radii, pack.radii)


class TestGeometricBias:
    def test_a_roomy_box_barely_biases_the_sizes(self) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0])
        bias = geometric_bias(spec, [500.0, 500.0])
        assert abs(bias["bias"]) < 0.01
        assert bias["excluded"] == 0.0
        assert bias["requested_mean"] == pytest.approx(2.0)

    def test_a_thin_slab_biases_the_sizes_hard(self) -> None:
        # This is why a "monolayer" of a wide distribution comes out
        # almost entirely fine: the tall particles have nowhere to go.
        spec = RadiusSpec.discrete([1.0, 3.0])
        bias = geometric_bias(spec, [100.0, 100.0, 6.2])
        assert bias["bias"] < -0.3
        assert bias["admitted_mean"] < bias["requested_mean"]

    def test_it_predicts_the_direction_of_the_realised_bias(self) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0])
        box = [120.0, 120.0, 6.5]
        predicted = geometric_bias(spec, box)["bias"]
        pack = sample(box, spec, seed=31)
        realised = float(pack.radii.mean() / 2.0 - 1.0)
        assert predicted < -0.2
        assert realised < -0.2

    def test_monodisperse_is_never_biased(self) -> None:
        bias = geometric_bias(RadiusSpec.discrete([2.0]), [100.0, 100.0, 4.1])
        assert bias["bias"] == pytest.approx(0.0)

    def test_reports_mass_that_cannot_fit_at_all(self) -> None:
        spec = RadiusSpec.discrete([1.0, 10.0], weights=[3.0, 1.0])
        bias = geometric_bias(spec, [100.0, 100.0, 5.0])
        assert bias["excluded"] == pytest.approx(0.25)

    def test_warns_when_geometry_drives_the_sizes(self, caplog) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0])
        with caplog.at_level("WARNING", logger="polydisc.sampling"):
            sample([120.0, 120.0, 6.5], spec, seed=32)
        assert "skews the size distribution" in caplog.text

    def test_stays_quiet_for_an_unremarkable_box(self, caplog) -> None:
        with caplog.at_level("WARNING", logger="polydisc.sampling"):
            sample([200.0, 200.0], RadiusSpec.discrete([1.0, 3.0]), seed=33)
        assert "skews the size distribution" not in caplog.text


class TestGenerationBounds:
    def test_fit_mode_reports_the_shrunken_box(self) -> None:
        pack = sample([100.0, 100.0], 3.0, seed=34, spacing=2.0)
        np.testing.assert_allclose(pack.generation_bounds, [50.0, 50.0])

    def test_expand_mode_reports_the_requested_box(self) -> None:
        pack = sample([100.0, 100.0], 3.0, seed=35, spacing=2.0, fit_bounds=False)
        np.testing.assert_allclose(pack.generation_bounds, [100.0, 100.0])
