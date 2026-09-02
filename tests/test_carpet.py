"""Tests for :mod:`polydisc.carpet`."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_inside, assert_no_overlaps

from polydisc.carpet import carpet, monolayer_thickness
from polydisc.radii import RadiusSpec
from polydisc.spacing import spacing_scale


class TestCarpetGeometry:
    def test_carpet_is_a_valid_thin_slab_packing(self) -> None:
        pack = carpet((80.0, 60.0), thickness=5.0, radii=2.0, seed=0)
        assert pack.dimensions == 3
        np.testing.assert_allclose(pack.bounds, [80.0, 60.0, 5.0])
        assert_inside(pack.points, pack.radii, pack.bounds)
        assert_no_overlaps(pack.points, pack.radii)

    def test_particles_are_confined_to_the_slab(self) -> None:
        pack = carpet((80.0, 80.0), thickness=6.0, radii=2.0, seed=1)
        assert np.all(pack.points[:, 2] >= 2.0 - 1e-12)
        assert np.all(pack.points[:, 2] <= 4.0 + 1e-12)

    def test_a_thinner_slab_gives_a_flatter_carpet(self) -> None:
        thin = carpet((80.0, 80.0), thickness=4.2, radii=2.0, seed=2)
        thick = carpet((80.0, 80.0), thickness=20.0, radii=2.0, seed=2)
        assert np.ptp(thin.points[:, 2]) < np.ptp(thick.points[:, 2])

    def test_monolayer_admits_the_largest_particle(self) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0])
        pack = carpet(
            (100.0, 100.0), thickness=monolayer_thickness(spec), radii=spec, seed=3
        )
        assert pack.radii.max() == 3.0

    def test_projected_area_can_exceed_the_fill_fraction(self) -> None:
        from polydisc.diagnostics import projected_area_fraction

        pack = carpet((100.0, 100.0), thickness=4.2, radii=2.0, seed=4)
        projected = projected_area_fraction(pack.radii, pack.bounds)
        assert projected > pack.fill_fraction


class TestMonolayerThickness:
    def test_is_just_above_the_largest_diameter(self) -> None:
        thickness = monolayer_thickness([1.0, 3.0])
        assert thickness > 6.0
        assert thickness == pytest.approx(6.0 * 1.02)

    def test_accounts_for_the_spacing_pre_shrink(self) -> None:
        spec = RadiusSpec.discrete([1.0, 3.0])
        spacing = 2.0
        thickness = monolayer_thickness(spec, spacing=spacing)
        assert thickness == pytest.approx(
            monolayer_thickness(spec) * spacing_scale(spacing)
        )
        # And the point of that inflation: the carpet must build.
        pack = carpet(
            (80.0, 80.0), thickness=thickness, radii=spec, spacing=spacing, seed=5
        )
        assert pack.n > 0
        assert pack.bounds[2] == pytest.approx(thickness)

    def test_no_inflation_when_bounds_are_not_fitted(self) -> None:
        spec = RadiusSpec.discrete([2.0])
        assert monolayer_thickness(
            spec, spacing=3.0, fit_bounds=False
        ) == pytest.approx(monolayer_thickness(spec))

    def test_rejects_bad_clearance(self) -> None:
        with pytest.raises(ValueError, match="clearance"):
            monolayer_thickness([1.0], clearance=0.0)


class TestCarpetValidation:
    @pytest.mark.parametrize("extent", [[10.0], [10.0, 10.0, 10.0]])
    def test_extent_must_be_two_dimensional(self, extent) -> None:
        with pytest.raises(ValueError, match="exactly 2 entries"):
            carpet(extent, thickness=5.0, radii=1.0)

    @pytest.mark.parametrize("thickness", [0.0, -1.0, np.nan])
    def test_thickness_must_be_positive(self, thickness) -> None:
        with pytest.raises(ValueError, match="thickness"):
            carpet((50.0, 50.0), thickness=thickness, radii=1.0)

    def test_slab_thinner_than_a_particle_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            carpet((50.0, 50.0), thickness=2.0, radii=2.0)
