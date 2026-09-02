"""Tests for :mod:`polydisc.spacing` and its effect on packings."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_inside, assert_no_overlaps, brute_force_min_gaps

from polydisc.sampling import sample
from polydisc.spacing import apply_spacing, expected_gap, spacing_scale


class TestScaleFactor:
    @pytest.mark.parametrize(
        ("spacing", "scale"), [(0.0, 1.0), (1.0, 1.5), (2.0, 2.0), (4.0, 3.0)]
    )
    def test_scale_is_one_plus_half_spacing(self, spacing, scale) -> None:
        assert spacing_scale(spacing) == pytest.approx(scale)

    @pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
    def test_rejects_bad_spacing(self, bad) -> None:
        with pytest.raises(ValueError):
            spacing_scale(bad)

    def test_apply_spacing_leaves_the_input_alone(self) -> None:
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaled = apply_spacing(points, 2.0)
        np.testing.assert_array_equal(points, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(scaled, [[2.0, 4.0], [6.0, 8.0]])


class TestGapSemantics:
    def test_two_touching_equal_particles_open_by_spacing_times_radius(
        self,
    ) -> None:
        # The defining property: spacing counts mean radii of clearance,
        # so spacing=2 fits one average particle between neighbours.
        radius = 3.0
        points = np.array([[0.0, 0.0], [2.0 * radius, 0.0]])

        for spacing in (0.5, 1.0, 2.0, 5.0):
            moved = apply_spacing(points, spacing)
            gap = float(np.linalg.norm(moved[1] - moved[0]) - 2.0 * radius)
            assert gap == pytest.approx(expected_gap(spacing, radius))
            assert gap == pytest.approx(spacing * radius)

    def test_one_average_particle_fits_at_spacing_two(self) -> None:
        radius = 2.5
        points = np.array([[0.0, 0.0, 0.0], [2.0 * radius, 0.0, 0.0]])
        moved = apply_spacing(points, 2.0)
        gap = float(np.linalg.norm(moved[1] - moved[0]) - 2.0 * radius)
        assert gap == pytest.approx(2.0 * radius)

    def test_unequal_pair_opens_by_half_spacing_times_contact_distance(
        self,
    ) -> None:
        r_i, r_j = 1.0, 4.0
        points = np.array([[0.0, 0.0], [r_i + r_j, 0.0]])
        moved = apply_spacing(points, 3.0)
        gap = float(np.linalg.norm(moved[1] - moved[0]) - (r_i + r_j))
        assert gap == pytest.approx(1.5 * (r_i + r_j))


class TestSamplingWithSpacing:
    def test_expand_mode_is_exactly_a_dilation_of_the_unspaced_run(self) -> None:
        # fit_bounds=False feeds the kernel identical arguments, so the
        # only difference must be the scale factor itself.
        plain = sample([100.0, 100.0], [2.0, 4.0], seed=17)
        spaced = sample(
            [100.0, 100.0], [2.0, 4.0], seed=17, spacing=3.0, fit_bounds=False
        )
        np.testing.assert_allclose(spaced.points, plain.points * 2.5, rtol=1e-12)
        np.testing.assert_array_equal(spaced.radii, plain.radii)
        np.testing.assert_allclose(spaced.bounds, [250.0, 250.0])

    def test_fit_mode_keeps_the_requested_extent(self) -> None:
        pack = sample([100.0, 100.0], 3.0, seed=18, spacing=2.0)
        np.testing.assert_allclose(pack.bounds, [100.0, 100.0])
        assert_inside(pack.points, pack.radii, np.array([100.0, 100.0]))

    def test_radii_are_never_scaled(self) -> None:
        plain = sample([100.0, 100.0], [2.0, 5.0], seed=19)
        spaced = sample(
            [100.0, 100.0], [2.0, 5.0], seed=19, spacing=4.0, fit_bounds=False
        )
        assert set(np.unique(spaced.radii)) <= {2.0, 5.0}
        np.testing.assert_array_equal(np.unique(plain.radii), np.unique(spaced.radii))

    def test_spacing_opens_a_real_gap_everywhere(self) -> None:
        spacing = 2.0
        pack = sample([120.0, 120.0], [2.0, 4.0], seed=20, spacing=spacing)
        gaps = brute_force_min_gaps(pack.points, pack.radii)

        assert_no_overlaps(pack.points, pack.radii)
        # Every particle clears its closest neighbour by at least the
        # spacing applied to the smallest possible pair.
        smallest_pair = 2.0 * float(pack.radii.min())
        assert gaps.min() >= 0.5 * spacing * smallest_pair - 1e-9

    def test_spacing_thins_the_packing_when_bounds_are_kept(self) -> None:
        plain = sample([150.0, 150.0], 3.0, seed=21)
        spaced = sample([150.0, 150.0], 3.0, seed=21, spacing=2.0)
        assert spaced.n < plain.n
        assert spaced.fill_fraction < plain.fill_fraction

    def test_zero_spacing_is_a_no_op(self) -> None:
        plain = sample([80.0, 80.0], 3.0, seed=22)
        spaced = sample([80.0, 80.0], 3.0, seed=22, spacing=0.0)
        np.testing.assert_array_equal(plain.points, spaced.points)
        assert spaced.scale == 1.0

    def test_scale_and_spacing_are_recorded(self) -> None:
        pack = sample([80.0, 80.0], 3.0, seed=23, spacing=1.0)
        assert pack.spacing == 1.0
        assert pack.scale == pytest.approx(1.5)
        assert pack.fit_bounds is True
