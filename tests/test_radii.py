"""Tests for :mod:`polydisc.radii`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from polydisc.radii import (
    KIND_DISCRETE,
    KIND_LOGNORMAL,
    KIND_NORMAL,
    KIND_UNIFORM,
    RadiusSpec,
    as_radius_spec,
)


class TestDiscrete:
    def test_uniform_weights_by_default(self) -> None:
        spec = RadiusSpec.discrete([1.0, 2.0, 3.0])
        assert spec.kind == KIND_DISCRETE
        assert spec.rmin == 1.0
        assert spec.rmax == 3.0
        np.testing.assert_allclose(spec.cdf, [1 / 3, 2 / 3, 1.0])

    def test_weights_are_normalised(self) -> None:
        spec = RadiusSpec.discrete([1.0, 2.0], weights=[3.0, 1.0])
        np.testing.assert_allclose(spec.cdf, [0.75, 1.0])

    def test_draws_match_requested_weights(self) -> None:
        spec = RadiusSpec.discrete([1.0, 2.0], weights=[1.0, 3.0])
        draws = spec.sample(100_000, seed=0)
        assert math.isclose(float(np.mean(draws == 2.0)), 0.75, abs_tol=0.01)

    def test_zero_weight_value_is_never_drawn(self) -> None:
        spec = RadiusSpec.discrete([1.0, 2.0, 3.0], weights=[1.0, 0.0, 1.0])
        draws = spec.sample(20_000, seed=1)
        assert not np.any(draws == 2.0)

    @pytest.mark.parametrize(
        ("values", "weights", "match"),
        [
            ([], None, "at least one value"),
            ([1.0, -2.0], None, "must all be > 0"),
            ([1.0, np.inf], None, "must all be finite"),
            ([1.0, 2.0], [1.0], "does not match"),
            ([1.0, 2.0], [1.0, -1.0], "non-negative"),
            ([1.0, 2.0], [0.0, 0.0], "sum to > 0"),
        ],
    )
    def test_rejects_bad_input(self, values, weights, match) -> None:
        with pytest.raises(ValueError, match=match):
            RadiusSpec.discrete(values, weights)


class TestLognormal:
    def test_default_truncation_is_two_gsd(self) -> None:
        spec = RadiusSpec.lognormal(10.0, 1.5)
        assert spec.kind == KIND_LOGNORMAL
        assert math.isclose(spec.rmin, 10.0 / 1.5**2)
        assert math.isclose(spec.rmax, 10.0 * 1.5**2)

    def test_draws_stay_inside_the_window(self) -> None:
        spec = RadiusSpec.lognormal(2.0, 1.6)
        draws = spec.sample(50_000, seed=3)
        assert draws.min() >= spec.rmin
        assert draws.max() <= spec.rmax

    def test_geometric_mean_is_preserved_by_truncation(self) -> None:
        spec = RadiusSpec.lognormal(2.0, 1.5)
        draws = spec.sample(200_000, seed=4)
        assert math.isclose(float(np.exp(np.mean(np.log(draws)))), 2.0, rel_tol=0.01)

    def test_truncation_shrinks_the_geometric_spread(self) -> None:
        # Truncating at +/- 2 GSD lowers the realised spread to about
        # gstd ** 0.88 -- the same relation PyFracVAL documents.
        spec = RadiusSpec.lognormal(1.0, 1.5)
        draws = spec.sample(200_000, seed=5)
        realised = float(np.exp(np.std(np.log(draws))))
        assert math.isclose(realised, 1.5**0.88, rel_tol=0.02)

    def test_unit_gstd_collapses_to_monodisperse(self) -> None:
        spec = RadiusSpec.lognormal(4.0, 1.0)
        assert spec.kind == KIND_DISCRETE
        assert spec.rmin == spec.rmax == 4.0

    def test_explicit_bounds_override_truncate(self) -> None:
        spec = RadiusSpec.lognormal(10.0, 1.5, rmin=8.0, rmax=12.0)
        assert (spec.rmin, spec.rmax) == (8.0, 12.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"gmean": 0.0, "gstd": 1.5}, "gmean"),
            ({"gmean": 1.0, "gstd": 0.5}, "gstd"),
            ({"gmean": 1.0, "gstd": 1.5, "truncate": 0.0}, "no finite rmax"),
        ],
    )
    def test_rejects_bad_input(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            RadiusSpec.lognormal(**kwargs)


class TestNormalAndUniform:
    def test_normal_truncates_at_two_sigma(self) -> None:
        spec = RadiusSpec.normal(10.0, 2.0)
        assert spec.kind == KIND_NORMAL
        assert (spec.rmin, spec.rmax) == (6.0, 14.0)

    def test_normal_draws_are_centred(self) -> None:
        spec = RadiusSpec.normal(10.0, 2.0)
        draws = spec.sample(100_000, seed=6)
        assert math.isclose(float(np.mean(draws)), 10.0, rel_tol=0.01)
        assert draws.min() >= 6.0 and draws.max() <= 14.0

    def test_zero_std_collapses_to_monodisperse(self) -> None:
        spec = RadiusSpec.normal(3.0, 0.0)
        assert spec.kind == KIND_DISCRETE

    def test_normal_rejects_a_window_reaching_zero(self) -> None:
        with pytest.raises(ValueError, match="rmin must be > 0"):
            RadiusSpec.normal(1.0, 1.0)

    def test_uniform_is_flat(self) -> None:
        spec = RadiusSpec.uniform(1.0, 3.0)
        assert spec.kind == KIND_UNIFORM
        draws = spec.sample(100_000, seed=7)
        assert math.isclose(float(np.mean(draws)), 2.0, rel_tol=0.01)
        assert draws.min() >= 1.0 and draws.max() <= 3.0

    def test_uniform_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="must be >= rmin"):
            RadiusSpec.uniform(3.0, 1.0)


class TestCdf:
    @pytest.mark.parametrize(
        "spec",
        [
            RadiusSpec.discrete([1.0, 2.0, 4.0], [1.0, 2.0, 1.0]),
            RadiusSpec.lognormal(2.0, 1.4),
            RadiusSpec.normal(5.0, 1.0),
            RadiusSpec.uniform(1.0, 2.0),
        ],
        ids=["discrete", "lognormal", "normal", "uniform"],
    )
    def test_cdf_spans_zero_to_one_and_is_monotone(self, spec) -> None:
        grid = np.linspace(spec.rmin, spec.rmax, 257)
        values = spec.cdf_at(grid)
        assert values[0] >= 0.0 and math.isclose(values[-1], 1.0, abs_tol=1e-9)
        assert np.all(np.diff(values) >= -1e-12)

    @pytest.mark.parametrize(
        "spec",
        [
            RadiusSpec.lognormal(2.0, 1.4),
            RadiusSpec.normal(5.0, 1.0),
            RadiusSpec.uniform(1.0, 2.0),
        ],
        ids=["lognormal", "normal", "uniform"],
    )
    def test_cdf_matches_the_empirical_draws(self, spec) -> None:
        draws = np.sort(spec.sample(40_000, seed=11))
        empirical = np.arange(1, draws.size + 1) / draws.size
        ks = float(np.max(np.abs(empirical - spec.cdf_at(draws))))
        assert ks < 0.02


class TestCoercion:
    def test_scalar_becomes_monodisperse(self) -> None:
        spec = as_radius_spec(3.0)
        assert spec.values.tolist() == [3.0]

    def test_sequence_becomes_discrete(self) -> None:
        spec = as_radius_spec([1.0, 2.0], weights=[1.0, 1.0])
        assert spec.kind == KIND_DISCRETE

    def test_existing_spec_passes_through(self) -> None:
        spec = RadiusSpec.uniform(1.0, 2.0)
        assert as_radius_spec(spec) is spec

    def test_weights_with_a_spec_are_rejected(self) -> None:
        spec = RadiusSpec.uniform(1.0, 2.0)
        with pytest.raises(ValueError, match="cannot be combined"):
            as_radius_spec(spec, weights=[1.0])

    def test_callable_gets_a_pointed_error(self) -> None:
        with pytest.raises(TypeError, match="RadiusSpec.lognormal"):
            as_radius_spec(lambda: 1.0)

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a RadiusSpec"):
            as_radius_spec({"r": 1})


class TestDescribe:
    def test_discrete_roundtrips_weights(self) -> None:
        spec = RadiusSpec.discrete([1.0, 2.0], [1.0, 3.0])
        described = spec.describe()
        assert described["kind"] == "discrete"
        np.testing.assert_allclose(described["weights"], [0.25, 0.75])

    def test_lognormal_reports_geometric_parameters(self) -> None:
        described = RadiusSpec.lognormal(2.0, 1.5).describe()
        assert described["kind"] == "lognormal"
        assert math.isclose(described["gmean"], 2.0)
        assert math.isclose(described["gstd"], 1.5)

    def test_sample_rejects_negative_size(self) -> None:
        with pytest.raises(ValueError, match="size must be >= 0"):
            RadiusSpec.uniform(1.0, 2.0).sample(-1)
