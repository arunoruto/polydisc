"""Tests for :mod:`polydisc.regions` and sampling inside each shape."""

from __future__ import annotations

import math

import numpy as np
import pytest
from packing_asserts import assert_no_overlaps

from polydisc.radii import RadiusSpec
from polydisc.regions import (
    Ball,
    Box,
    as_region,
    region_from_metadata,
)
from polydisc.sampling import geometric_bias, sample


class TestBox:
    def test_properties(self) -> None:
        box = Box([100.0, 50.0])
        assert box.dimensions == 2
        assert box.volume == pytest.approx(5000.0)
        assert box.footprint == pytest.approx(5000.0)
        assert box.max_particle_radius == pytest.approx(25.0)
        np.testing.assert_allclose(box.bounding_box, [100.0, 50.0])

    def test_three_dimensional_footprint_ignores_depth(self) -> None:
        assert Box([10.0, 20.0, 3.0]).footprint == pytest.approx(200.0)
        assert Box([10.0, 20.0, 3.0]).volume == pytest.approx(600.0)

    def test_scaled(self) -> None:
        np.testing.assert_allclose(Box([10.0, 20.0]).scaled(2.5).bounds, [25.0, 50.0])

    def test_contains(self) -> None:
        box = Box([10.0, 10.0])
        points = np.array([[5.0, 5.0], [1.0, 5.0], [0.5, 5.0]])
        radii = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(box.contains(points, radii), [True, True, False])

    def test_bounding_box_is_a_copy(self) -> None:
        box = Box([10.0, 10.0])
        box.bounding_box[0] = 999.0
        assert box.bounds[0] == 10.0

    @pytest.mark.parametrize(
        ("bounds", "match"),
        [
            ([10.0], "2 or 3 entries"),
            ([1.0, 2.0, 3.0, 4.0], "2 or 3 entries"),
            ([10.0, 0.0], "must all be > 0"),
            ([10.0, np.inf], "must all be finite"),
        ],
    )
    def test_validation(self, bounds, match) -> None:
        with pytest.raises(ValueError, match=match):
            Box(bounds)


class TestBall:
    def test_properties_3d(self) -> None:
        ball = Ball(10.0)
        assert ball.dimensions == 3
        assert ball.volume == pytest.approx(4.0 / 3.0 * math.pi * 1000.0)
        assert ball.footprint == pytest.approx(math.pi * 100.0)
        assert ball.max_particle_radius == pytest.approx(10.0)
        np.testing.assert_allclose(ball.bounding_box, [20.0, 20.0, 20.0])
        np.testing.assert_allclose(ball.centre, [10.0, 10.0, 10.0])

    def test_properties_2d(self) -> None:
        ball = Ball(10.0, dimensions=2)
        assert ball.volume == pytest.approx(math.pi * 100.0)
        np.testing.assert_allclose(ball.bounding_box, [20.0, 20.0])

    def test_scaled(self) -> None:
        scaled = Ball(10.0, dimensions=2).scaled(3.0)
        assert scaled.radius == pytest.approx(30.0)
        assert scaled.dimensions == 2

    def test_contains(self) -> None:
        ball = Ball(10.0, dimensions=2)
        points = np.array([[10.0, 10.0], [19.0, 10.0], [19.5, 10.0]])
        radii = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(ball.contains(points, radii), [True, True, False])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"radius": 0.0}, "radius"),
            ({"radius": -1.0}, "radius"),
            ({"radius": np.nan}, "radius"),
            ({"radius": 1.0, "dimensions": 4}, "dimensions"),
        ],
    )
    def test_validation(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            Ball(**kwargs)


class TestAdmissibleMeasure:
    def test_box_is_the_shrunken_box(self) -> None:
        measure = Box([100.0, 100.0]).admissible_measure([1.0, 3.0])
        np.testing.assert_allclose(measure, [98.0**2, 94.0**2])

    def test_ball_is_the_shrunken_ball(self) -> None:
        measure = Ball(50.0).admissible_measure([1.0, 3.0])
        np.testing.assert_allclose(measure, [49.0**3, 47.0**3])

    def test_zero_where_the_particle_cannot_fit(self) -> None:
        assert Ball(2.0).admissible_measure([5.0])[0] == 0.0
        assert Box([4.0, 4.0]).admissible_measure([5.0])[0] == 0.0

    def test_ball_matches_its_bounding_cube_exactly(self) -> None:
        # Both admit a fraction (1 - r/R)^d of their own measure, so the
        # predicted size bias is identical -- inscribing a ball in a cube
        # changes the capacity but not the size mix.
        spec = RadiusSpec.discrete([1.0, 8.0])
        ball = Ball(20.0)
        cube = Box(np.full(3, 2.0 * ball.radius))
        assert geometric_bias(spec, ball)["bias"] == pytest.approx(
            geometric_bias(spec, cube)["bias"]
        )

    def test_an_equal_volume_box_biases_sizes_harder_than_the_ball(self) -> None:
        # A cube of the same volume has a smaller inradius than the ball,
        # so it is the box that squeezes out large particles.
        spec = RadiusSpec.discrete([1.0, 8.0])
        ball = Ball(20.0)
        box = Box(np.full(3, ball.volume ** (1.0 / 3.0)))
        assert geometric_bias(spec, box)["bias"] < geometric_bias(spec, ball)["bias"]


class TestCoercionAndMetadata:
    def test_sequence_becomes_a_box(self) -> None:
        assert as_region([10.0, 20.0]).describe()["shape"] == "box"

    def test_region_passes_through(self) -> None:
        ball = Ball(3.0)
        assert as_region(ball) is ball

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a Box, a Ball"):
            as_region("big")

    @pytest.mark.parametrize(
        "region", [Box([10.0, 20.0]), Ball(5.0), Ball(5.0, dimensions=2)]
    )
    def test_describe_roundtrips(self, region) -> None:
        restored = region_from_metadata(region.describe())
        assert restored == region

    @pytest.mark.parametrize("meta", [None, {}, {"shape": "torus"}, {"shape": "ball"}])
    def test_unusable_metadata_returns_none(self, meta) -> None:
        assert region_from_metadata(meta) is None


class TestSamplingInABall:
    @pytest.mark.parametrize(
        ("region", "radii"),
        [
            (Ball(30.0, dimensions=2), 2.0),
            (Ball(30.0, dimensions=2), [1.5, 3.0]),
            (Ball(25.0), 3.0),
            (Ball(25.0), RadiusSpec.uniform(2.0, 4.0)),
        ],
        ids=["2d-mono", "2d-poly", "3d-mono", "3d-uniform"],
    )
    def test_every_particle_is_inside_and_disjoint(self, region, radii) -> None:
        pack = sample(region, radii, repetitions=15, seed=0)
        assert pack.n > 0
        assert np.all(pack.region.contains(pack.points, pack.radii))
        assert_no_overlaps(pack.points, pack.radii)

    def test_no_particle_sits_in_the_bounding_box_corners(self) -> None:
        # A box sampler would happily fill the corners; a ball must not.
        ball = Ball(30.0, dimensions=2)
        pack = sample(ball, 2.0, seed=1)
        distance = np.linalg.norm(pack.points - ball.centre[None, :], axis=1)
        assert distance.max() <= ball.radius

    def test_fill_fraction_uses_the_ball_volume(self) -> None:
        pack = sample(Ball(25.0), 3.0, seed=2)
        expected = float(np.sum(4.0 / 3.0 * np.pi * pack.radii**3)) / pack.region.volume
        assert pack.fill_fraction == pytest.approx(expected)
        assert 0.0 < pack.fill_fraction < 1.0

    def test_determinism(self) -> None:
        first = sample(Ball(20.0), 2.0, seed=5)
        second = sample(Ball(20.0), 2.0, seed=5)
        np.testing.assert_array_equal(first.points, second.points)

    def test_spacing_keeps_particles_inside(self) -> None:
        pack = sample(Ball(40.0), 3.0, seed=6, spacing=2.0)
        assert pack.region.radius == pytest.approx(40.0)
        assert np.all(pack.region.contains(pack.points, pack.radii))
        assert_no_overlaps(pack.points, pack.radii)

    def test_spacing_can_grow_the_ball_instead(self) -> None:
        pack = sample(Ball(40.0), 3.0, seed=6, spacing=2.0, fit_bounds=False)
        assert pack.region.radius == pytest.approx(80.0)
        assert np.all(pack.region.contains(pack.points, pack.radii))

    def test_particle_larger_than_the_ball_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            sample(Ball(2.0), 5.0)

    def test_a_ball_barely_larger_than_one_particle_holds_one(self) -> None:
        pack = sample(Ball(1.05), 1.0, seed=0)
        assert pack.n == 1
        assert np.all(pack.region.contains(pack.points, pack.radii))

    def test_save_load_keeps_the_shape(self, tmp_path) -> None:
        from polydisc.sampling import Packing

        pack = sample(Ball(20.0, dimensions=2), 2.0, seed=7)
        loaded = Packing.load(pack.save(tmp_path / "ball.npz"))
        assert loaded.region == pack.region
        assert loaded.fill_fraction == pytest.approx(pack.fill_fraction)


class TestCarpetRecipe:
    def test_a_thin_box_is_a_carpet(self) -> None:
        # No special support needed: a carpet is a box one particle deep.
        radius = 2.0
        pack = sample([120.0, 120.0, 2.05 * radius], radius, seed=0)
        assert pack.dimensions == 3
        assert np.all(pack.points[:, 2] >= radius - 1e-12)
        assert np.all(pack.points[:, 2] <= 1.05 * radius + 1e-12)
        assert_no_overlaps(pack.points, pack.radii)
        # Assert the carpet is actually full. Without this the test
        # passes on a packing of one particle, which is exactly what a
        # slab this thin used to produce.
        coverage = float(np.sum(np.pi * pack.radii**2) / (120.0 * 120.0))
        assert coverage > 0.4

    @pytest.mark.parametrize("radius", [0.5, 2.0, 5.0])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_monolayer_fills_whatever_the_seed(self, radius, seed) -> None:
        # A slab barely thicker than one particle admits only a sliver of
        # candidate directions. Charging the rest against Bridson's k
        # retired the first active point immediately and the run stopped
        # at a single particle, for every seed.
        extent = 60.0 * radius
        pack = sample([extent, extent, 2.05 * radius], radius, seed=seed)
        coverage = float(np.sum(np.pi * pack.radii**2) / extent**2)
        assert coverage > 0.4, f"carpet only reached {coverage:.3f} coverage"

    def test_a_thin_box_fills_on_any_axis(self) -> None:
        radius, extent = 2.0, 120.0
        depth = 2.05 * radius
        for axis in range(3):
            bounds = [extent, extent, extent]
            bounds[axis] = depth
            pack = sample(bounds, radius, seed=0)
            face = extent * extent
            assert float(np.sum(np.pi * pack.radii**2) / face) > 0.35

    def test_spacing_needs_the_depth_pre_inflated(self) -> None:
        from polydisc.spacing import spacing_scale

        radius, spacing = 2.0, 2.0
        depth = 2.05 * radius * spacing_scale(spacing)
        pack = sample([120.0, 120.0, depth], radius, spacing=spacing, seed=0)
        assert pack.n > 0
        # Without the inflation the pre-shrink leaves no room at all.
        with pytest.raises(ValueError, match="does not fit"):
            sample([120.0, 120.0, 2.05 * radius], radius, spacing=spacing, seed=0)

    def test_projected_area_exceeds_fill_fraction(self) -> None:
        from polydisc.diagnostics import projected_area_fraction

        pack = sample([120.0, 120.0, 4.2], 2.0, seed=0)
        projected = projected_area_fraction(pack.radii, pack.region)
        assert projected > pack.fill_fraction
