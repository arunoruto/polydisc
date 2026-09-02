"""Tests for the dashboard's importable pieces.

The Streamlit app itself is exercised through ``AppTest``; the loaders
and the deployment metadata are tested directly, since a broken
requirements file only shows up as a failed deploy otherwise.
"""

from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="dashboard extra not installed")
pytest.importorskip("plotly", reason="dashboard extra not installed")

import views
from loaders import (
    LoadError,
    infer_region,
    infer_spec,
    load_packing,
    packing_from_csv,
    packing_from_npz,
    shift_to_origin,
)

from polydisc.radii import KIND_DISCRETE, KIND_UNIFORM
from polydisc.sampling import sample

REPO = Path(__file__).resolve().parent.parent
DASHBOARD = REPO / "dashboard"


class TestRequirementsStayInStep:
    """The Cloud requirements file duplicates pyproject by necessity."""

    @staticmethod
    def _requirement_lines() -> set[str]:
        text = (DASHBOARD / "requirements.txt").read_text()
        return {
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    @staticmethod
    def _pyproject_requirements() -> set[str]:
        data = tomllib.loads((REPO / "pyproject.toml").read_text())
        project = data["project"]
        return set(project["dependencies"]) | set(
            project["optional-dependencies"]["dashboard"]
        )

    def test_requirements_match_pyproject_exactly(self) -> None:
        declared = self._pyproject_requirements()
        listed = self._requirement_lines()
        assert listed == declared, (
            "dashboard/requirements.txt has drifted from pyproject.toml.\n"
            f"  only in requirements.txt: {sorted(listed - declared)}\n"
            f"  only in pyproject.toml:   {sorted(declared - listed)}"
        )

    def test_streamlit_is_optional_not_a_core_dependency(self) -> None:
        data = tomllib.loads((REPO / "pyproject.toml").read_text())
        core = " ".join(data["project"]["dependencies"])
        extra = " ".join(data["project"]["optional-dependencies"]["dashboard"])
        assert "streamlit" not in core
        assert "streamlit" in extra

    def test_entrypoint_and_config_exist(self) -> None:
        assert (DASHBOARD / "app.py").is_file()
        assert (REPO / ".streamlit" / "config.toml").is_file()


class TestInference:
    def test_region_spans_the_particles(self) -> None:
        points = np.array([[1.0, 1.0], [4.0, 3.0]])
        region = infer_region(points, np.array([1.0, 1.0]))
        np.testing.assert_allclose(region.bounds, [5.0, 4.0])

    def test_shift_puts_the_low_corner_at_zero(self) -> None:
        points = np.array([[5.0, 7.0], [9.0, 7.0]])
        radii = np.array([1.0, 2.0])
        shifted = shift_to_origin(points, radii)
        np.testing.assert_allclose(np.min(shifted - radii[:, None], axis=0), [0.0, 0.0])

    def test_few_distinct_radii_infer_a_discrete_spec(self) -> None:
        spec = infer_spec(np.array([1.0, 1.0, 1.0, 2.0]))
        assert spec.kind == KIND_DISCRETE
        np.testing.assert_allclose(spec.values, [1.0, 2.0])
        np.testing.assert_allclose(spec.cdf, [0.75, 1.0])

    def test_many_distinct_radii_infer_a_uniform_spec(self) -> None:
        spec = infer_spec(np.linspace(1.0, 2.0, 200))
        assert spec.kind == KIND_UNIFORM
        assert (spec.rmin, spec.rmax) == (1.0, 2.0)


class TestCsvLoading:
    def test_two_dimensional(self) -> None:
        pack = packing_from_csv(b"x,y,r\n1,1,1\n4,1,1\n")
        assert (pack.n, pack.dimensions) == (2, 2)
        np.testing.assert_allclose(pack.points, [[1.0, 1.0], [4.0, 1.0]])

    def test_three_dimensional(self) -> None:
        pack = packing_from_csv(b"x,y,z,r\n1,1,1,1\n1,1,4,1\n")
        assert (pack.n, pack.dimensions) == (2, 3)

    def test_round_trips_a_generated_packing(self) -> None:
        original = sample([60.0, 60.0], [2.0, 3.0], seed=1)
        rows = "".join(
            f"{p[0]:.10g},{p[1]:.10g},{r:.10g}\n"
            for p, r in zip(original.points, original.radii)
        )
        pack = packing_from_csv(("x,y,r\n" + rows).encode())
        assert pack.n == original.n
        np.testing.assert_allclose(np.sort(pack.radii), np.sort(original.radii))

    def test_tolerates_blank_lines_and_a_bom(self) -> None:
        pack = packing_from_csv("﻿x,y,r\n1,1,1\n\n4,1,1\n\n".encode())
        assert pack.n == 2

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (b"", "empty"),
            (b"x,y,r\n", "no data rows"),
            (b"a,b,c\n1,2,3\n", "expected a header"),
            (b"x,y,r\n1,2\n", "expected 3 values"),
            (b"x,y,r\n1,2,oops\n", "line 2"),
            (b"x,y,r\n1,2,0\n", "must all be > 0"),
            (b"x,y,r\n1,2,nan\n", "finite"),
        ],
    )
    def test_rejects_bad_input(self, data, match) -> None:
        with pytest.raises(LoadError, match=match):
            packing_from_csv(data)


class TestNpzLoading:
    def test_round_trips_full_metadata(self, tmp_path) -> None:
        original = sample([80.0, 80.0, 10.0], [1.0, 2.0], seed=3, spacing=1.0)
        path = original.save(tmp_path / "pack.npz")

        pack = packing_from_npz(path.read_bytes())
        np.testing.assert_array_equal(pack.points, original.points)
        assert pack.region == original.region
        assert pack.seed == original.seed
        assert pack.spacing == 1.0

    def test_falls_back_for_a_foreign_archive(self) -> None:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            points=np.array([[0.0, 0.0], [4.0, 0.0]]),
            radii=np.array([1.0, 1.0]),
        )
        pack = packing_from_npz(buffer.getvalue())
        assert pack.n == 2
        np.testing.assert_allclose(pack.bounds, [6.0, 2.0])

    def test_rejects_an_archive_without_the_arrays(self) -> None:
        buffer = io.BytesIO()
        np.savez(buffer, something_else=np.zeros(3))
        with pytest.raises(LoadError, match="points"):
            packing_from_npz(buffer.getvalue())

    def test_rejects_a_file_that_is_not_an_archive(self) -> None:
        with pytest.raises(LoadError, match="not a .npz archive"):
            packing_from_npz(b"definitely not a zip file")


class TestDispatch:
    def test_unknown_suffix(self) -> None:
        with pytest.raises(LoadError, match="expected a .npz or .csv"):
            load_packing("pack.txt", b"")

    def test_suffix_is_case_insensitive(self) -> None:
        assert load_packing("PACK.CSV", b"x,y,r\n0,0,1\n3,0,1\n").n == 2


class TestViews:
    @pytest.fixture(scope="class")
    def packing_2d(self):
        return sample([80.0, 60.0], [2.0, 4.0], seed=0)

    @pytest.fixture(scope="class")
    def packing_3d(self):
        return sample([50.0, 50.0, 12.0], [1.0, 2.0], seed=0, spacing=1.0)

    def test_plot_2d(self, packing_2d) -> None:
        figure, reduced = views.plot_2d(packing_2d)
        assert not reduced
        assert figure.axes

    def test_plot_2d_subsamples_above_the_limit(self, packing_2d) -> None:
        _, reduced = views.plot_2d(packing_2d, limit=10)
        assert reduced

    def test_plot_3d(self, packing_3d) -> None:
        figure, _ = views.plot_3d(packing_3d)
        assert len(figure.data) == 1
        assert len(figure.data[0].x) == packing_3d.n

    def test_plot_slice_cuts_a_real_cross_section(self, packing_3d) -> None:
        figure, shown = views.plot_slice(packing_3d, 2, 6.0, 3.0)
        assert 0 < shown < packing_3d.n
        assert figure.axes

    def test_plot_slice_of_an_empty_slab_still_renders(self, packing_3d) -> None:
        figure, shown = views.plot_slice(packing_3d, 2, -100.0, 0.1)
        assert shown == 0
        assert figure.axes

    def test_radius_histogram_overlays_the_request(self, packing_2d) -> None:
        assert len(views.plot_radius_histogram(packing_2d, True).data) == 2
        assert len(views.plot_radius_histogram(packing_2d, False).data) == 1

    def test_gap_histogram_marks_the_spacing(self, packing_3d) -> None:
        figure = views.plot_gap_histogram(packing_3d)
        assert len(figure.data) == 1
        assert figure.layout.shapes  # the requested-spacing line

    def test_markers_are_big_enough_to_see(self, packing_3d) -> None:
        # Plotly marker size is a diameter in pixels, not data units.
        # Deriving it from radii without that conversion produced
        # sub-pixel markers and an apparently empty plot.
        sizes = np.asarray(views.plot_3d(packing_3d)[0].data[0].marker.size)
        assert sizes.min() >= 2.0
        assert np.median(sizes) > 3.0

    def test_markers_are_capped_for_a_tiny_container(self) -> None:
        pack = sample([12.0, 12.0, 12.0], 2.0, seed=0)
        sizes = np.asarray(views.plot_3d(pack)[0].data[0].marker.size)
        assert sizes.max() <= 40.0

    def test_marker_size_tracks_relative_radius(self) -> None:
        pack = sample([60.0, 60.0, 60.0], [1.0, 3.0], seed=0)
        figure, _ = views.plot_3d(pack)
        sizes = np.asarray(figure.data[0].marker.size)
        radii = np.asarray(figure.data[0].marker.color)
        assert sizes[radii.argmax()] > sizes[radii.argmin()]

    @pytest.mark.parametrize(
        "bounds",
        [[400.0, 20.0], [20.0, 400.0], [400.0, 400.0]],
        ids=["very-wide", "very-tall", "square"],
    )
    def test_figure_height_stays_in_a_readable_range(self, bounds) -> None:
        # A carpet cross-section is far wider than it is deep. Sizing the
        # figure at that true aspect leaves a sliver whose axis labels are
        # larger than the plot; the axes stay equal-aspect either way.
        figure, _ = views.plot_2d(sample(bounds, 2.0, seed=0))
        _, height = figure.get_size_inches()
        assert 2.5 <= height <= 11.0
        assert figure.axes[0].get_aspect() == 1.0

    def test_slice_of_a_thin_carpet_is_readable(self) -> None:
        pack = sample([120.0, 120.0, 4.1], 2.0, seed=0)
        assert pack.n > 100, "the carpet itself failed to fill"
        figure, shown = views.plot_slice(pack, 0, 60.0, 6.0)
        _, height = figure.get_size_inches()
        assert shown > 0
        assert height >= 2.5

    def test_subsample_is_deterministic(self, packing_2d) -> None:
        first, _, _ = views.subsample(packing_2d, 20)
        second, _, _ = views.subsample(packing_2d, 20)
        np.testing.assert_array_equal(first, second)


class TestApp:
    """End-to-end runs of the Streamlit script itself."""

    @staticmethod
    def _app():
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=300)

    def test_landing_page_renders(self) -> None:
        at = self._app().run()
        assert not at.exception
        assert any("Generate" in b.label for b in at.button)

    def test_generate_produces_a_packing(self) -> None:
        at = self._app().run()
        at.button[0].click().run()
        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert int(metrics["particles"].replace(",", "")) > 0
        assert metrics["overlaps"] == "0"
        assert len(at.tabs) == 5

    @pytest.mark.parametrize(
        "shape", ["2D box", "3D box", "2D ball", "3D ball", "carpet (thin box)"]
    )
    def test_every_container_generates(self, shape) -> None:
        at = self._app().run()
        at.selectbox[0].set_value(shape).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["overlaps"] == "0"

    @pytest.mark.parametrize("family", ["lognormal", "discrete", "normal", "uniform"])
    def test_every_distribution_generates(self, family) -> None:
        at = self._app().run()
        at.selectbox[1].set_value(family).run()
        at.button[0].click().run()
        assert not at.exception, at.exception

    def test_an_impossible_request_is_reported_not_raised(self) -> None:
        at = self._app().run()
        at.selectbox[0].set_value("2D box").run()
        # A 1-wide box cannot hold the default radii.
        at.number_input[0].set_value(1.0).run()
        at.number_input[1].set_value(1.0).run()
        at.button[0].click().run()
        assert not at.exception
        assert at.error
        assert "does not fit" in at.error[0].value

    def test_metadata_survives_a_download_reload(self, tmp_path) -> None:
        original = sample([60.0, 60.0], 3.0, seed=11, spacing=2.0)
        reloaded = packing_from_npz(original.save(tmp_path / "p.npz").read_bytes())
        assert json.loads(json.dumps(reloaded.metadata())) == json.loads(
            json.dumps(original.metadata())
        )
