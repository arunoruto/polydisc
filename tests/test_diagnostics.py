"""Tests for :mod:`polydisc.diagnostics`."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import brute_force_min_gaps

from polydisc.carpet import carpet
from polydisc.diagnostics import (
    gap_report,
    nearest_neighbour_gaps,
    projected_area_fraction,
    radius_report,
    report,
)
from polydisc.radii import RadiusSpec
from polydisc.sampling import sample


class TestNearestNeighbourGaps:
    def test_matches_a_hand_computed_pair(self) -> None:
        points = np.array([[0.0, 0.0], [5.0, 0.0]])
        radii = np.array([1.0, 2.0])
        np.testing.assert_allclose(nearest_neighbour_gaps(points, radii), [2.0, 2.0])

    @pytest.mark.parametrize(
        ("bounds", "radii"),
        [
            ([70.0, 70.0], [2.0, 5.0]),
            ([40.0, 40.0, 40.0], [2.0, 4.0]),
            ([70.0, 70.0], RadiusSpec.uniform(1.0, 4.0)),
        ],
        ids=["2d", "3d", "2d-wide"],
    )
    def test_matches_brute_force_on_a_real_packing(self, bounds, radii) -> None:
        pack = sample(bounds, radii, repetitions=12, seed=0)
        np.testing.assert_allclose(
            nearest_neighbour_gaps(pack.points, pack.radii),
            brute_force_min_gaps(pack.points, pack.radii),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_finds_a_neighbour_far_outside_the_search_cell(self) -> None:
        # Two tiny particles a long way apart: the shell search has to
        # keep expanding rather than report inf.
        points = np.array([[0.0, 0.0], [500.0, 0.0]])
        radii = np.array([0.5, 0.5])
        np.testing.assert_allclose(
            nearest_neighbour_gaps(points, radii), [499.0, 499.0]
        )

    def test_detects_a_deliberate_overlap(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        radii = np.array([1.0, 1.0])
        np.testing.assert_allclose(nearest_neighbour_gaps(points, radii), [-1.0, -1.0])

    @pytest.mark.parametrize("count", [0, 1])
    def test_degenerate_inputs_are_infinite(self, count) -> None:
        points = np.zeros((count, 2))
        gaps = nearest_neighbour_gaps(points, np.ones(count))
        assert gaps.shape == (count,)
        assert np.all(np.isinf(gaps))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            nearest_neighbour_gaps(np.zeros((3, 2)), np.ones(2))

    def test_rejects_unsupported_dimensions(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, 2\) or \(N, 3\)"):
            nearest_neighbour_gaps(np.zeros((3, 4)), np.ones(3))


class TestGapReport:
    def test_reports_no_overlaps_for_a_valid_packing(self) -> None:
        pack = sample([80.0, 80.0], [2.0, 4.0], seed=1)
        stats = gap_report(pack.points, pack.radii)
        assert stats["overlaps"] == 0
        assert stats["worst_overlap"] == 0.0
        assert stats["min"] <= stats["median"] <= stats["p95"]

    def test_counts_a_deliberate_overlap(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 50.0]])
        radii = np.array([1.0, 1.0, 1.0])
        stats = gap_report(points, radii)
        assert stats["overlaps"] == 2
        assert stats["worst_overlap"] == pytest.approx(1.0)

    def test_gap_over_radius_tracks_the_requested_spacing(self) -> None:
        spacing = 2.0
        pack = sample([150.0, 150.0], 3.0, seed=2, spacing=spacing)
        stats = gap_report(pack.points, pack.radii)
        # Monodisperse and saturated, so the closest pair was touching
        # and its gap is exactly spacing * r after the dilation.
        assert stats["min_over_r"] == pytest.approx(spacing, rel=0.05)


class TestRadiusReport:
    def test_reports_realised_moments(self) -> None:
        radii = np.array([1.0, 2.0, 3.0, 4.0])
        stats = radius_report(radii)
        assert stats["n"] == 4
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == 1.0 and stats["max"] == 4.0
        assert stats["gmean"] == pytest.approx(float(np.exp(np.mean(np.log(radii)))))

    def test_requested_mean_for_a_discrete_spec_is_exact(self) -> None:
        spec = RadiusSpec.discrete([2.0, 4.0, 6.0])
        stats = radius_report(spec.sample(100, seed=0), spec)
        assert stats["requested_mean"] == pytest.approx(4.0)

    def test_requested_mean_for_a_weighted_spec_is_exact(self) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0], weights=[3.0, 1.0])
        stats = radius_report(np.array([1.0]), spec)
        assert stats["requested_mean"] == pytest.approx(1.5)

    def test_unbiased_draws_report_a_small_bias(self) -> None:
        # Straight from the distribution, with no placement involved:
        # bias and KS must both be near zero, which pins the reference
        # side of the comparison.
        spec = RadiusSpec.lognormal(2.0, 1.4)
        stats = radius_report(spec.sample(50_000, seed=3), spec)
        assert abs(stats["mean_bias"]) < 0.02
        assert stats["ks"] < 0.02

    def test_placement_bias_is_visible_in_a_saturated_packing(self) -> None:
        # The dart-throwing bias is real and the report must show it
        # rather than silently pass the request through.
        spec = RadiusSpec.discrete([1.0, 5.0])
        pack = sample([150.0, 150.0], spec, seed=4)
        stats = radius_report(pack.radii, spec)
        assert stats["mean_bias"] < -0.05

    def test_empty_input(self) -> None:
        assert radius_report(np.empty(0)) == {"n": 0}


class TestProjectedArea:
    def test_matches_a_hand_computation(self) -> None:
        radii = np.array([1.0, 2.0])
        expected = (np.pi * 1.0 + np.pi * 4.0) / (10.0 * 20.0)
        assert projected_area_fraction(radii, [10.0, 20.0]) == pytest.approx(expected)

    def test_ignores_the_third_bound(self) -> None:
        radii = np.array([1.0])
        assert projected_area_fraction(radii, [10.0, 10.0, 3.0]) == pytest.approx(
            projected_area_fraction(radii, [10.0, 10.0])
        )

    def test_degenerate_footprint_is_zero(self) -> None:
        assert projected_area_fraction(np.array([1.0]), [0.0, 10.0]) == 0.0


class TestReport:
    def test_full_report_of_a_carpet(self) -> None:
        pack = carpet((80.0, 80.0), thickness=9.0, radii=2.0, seed=5, spacing=1.0)
        rep = report(pack)

        assert rep.n == pack.n
        assert rep.dimensions == 3
        assert rep.spacing == 1.0
        assert rep.gaps["overlaps"] == 0
        assert rep.projected_area_fraction > 0.0
        assert math.isclose(rep.fill_fraction, pack.fill_fraction)

    def test_two_dimensional_report_has_no_projection(self) -> None:
        rep = report(sample([60.0, 60.0], 3.0, seed=6))
        assert rep.projected_area_fraction == 0.0
        assert "projected area" not in str(rep)

    def test_text_summary_mentions_the_essentials(self) -> None:
        text = str(report(sample([60.0, 60.0], 3.0, seed=7, spacing=1.0)))
        assert "particles" in text
        assert "fill fraction" in text
        assert "neighbour gap" in text
        assert "spacing requested 1" in text

    def test_report_is_json_serialisable(self) -> None:
        import json

        rep = report(sample([60.0, 60.0], [2.0, 3.0], seed=8))
        assert json.loads(json.dumps(rep.to_dict()))["n"] == rep.n

    def test_unsaturated_run_is_flagged(self) -> None:
        rep = report(sample([200.0, 200.0], 3.0, seed=9, max_points=25))
        assert not rep.saturated
        assert "stopped at max_points" in str(rep)
