"""Tests for :mod:`polydisc.cli`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from polydisc.cli import cli
from polydisc.sampling import Packing


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _csv_rows(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()]


class TestGenerate:
    def test_prints_csv_to_stdout(self, runner) -> None:
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "60,60",
                "--radii",
                "3,5",
                "--seed",
                "0",
                "--no-report",
            ],
        )
        assert result.exit_code == 0, result.output
        rows = _csv_rows(result.output)
        assert rows[0] == "x,y,r"
        assert len(rows) > 1

    def test_three_dimensional_header(self, runner) -> None:
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "40,40,40",
                "--radii",
                "4",
                "--seed",
                "0",
                "--no-report",
            ],
        )
        assert result.exit_code == 0, result.output
        assert _csv_rows(result.output)[0] == "x,y,z,r"

    def test_seed_makes_output_reproducible(self, runner) -> None:
        args = [
            "generate",
            "--bounds",
            "60,60",
            "--radii",
            "3",
            "--seed",
            "7",
            "--no-report",
        ]
        first = runner.invoke(cli, args)
        second = runner.invoke(cli, args)
        assert first.exit_code == 0, first.output
        assert first.output == second.output

    def test_writes_npz(self, runner, tmp_path) -> None:
        out = tmp_path / "pack.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "60,60",
                "--radii",
                "3",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        assert pack.n > 0
        assert pack.seed == 0

    def test_writes_csv_creating_parents(self, runner, tmp_path) -> None:
        out = tmp_path / "nested" / "pack.csv"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "50,50",
                "--radii",
                "4",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.read_text().startswith("x,y,r")

    def test_lognormal_distribution(self, runner, tmp_path) -> None:
        out = tmp_path / "pack.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "80,80",
                "--dist",
                "lognormal",
                "--gmean",
                "2",
                "--gstd",
                "1.4",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        assert pack.spec.describe()["kind"] == "lognormal"
        assert pack.radii.min() >= 2.0 / 1.4**2 - 1e-9

    def test_spacing_thins_the_packing(self, runner, tmp_path) -> None:
        counts = []
        for spacing in ("0", "2"):
            out = tmp_path / f"pack{spacing}.npz"
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--bounds",
                    "120,120",
                    "--radii",
                    "3",
                    "--spacing",
                    spacing,
                    "--seed",
                    "0",
                    "--out",
                    str(out),
                ],
            )
            assert result.exit_code == 0, result.output
            counts.append(Packing.load(out).n)
        assert counts[1] < counts[0]

    def test_report_goes_to_stderr_not_stdout(self, runner) -> None:
        result = runner.invoke(
            cli,
            ["generate", "--bounds", "50,50", "--radii", "4", "--seed", "0"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Mixing diagnostics into piped CSV would corrupt it.
        assert _csv_rows(result.stdout)[0] == "x,y,r"
        assert "fill fraction" in result.stderr


class TestGenerateErrors:
    def test_missing_bounds(self, runner) -> None:
        result = runner.invoke(cli, ["generate", "--radii", "3"])
        assert result.exit_code != 0
        assert "--bounds is required" in result.output

    def test_bad_bounds_length(self, runner) -> None:
        result = runner.invoke(cli, ["generate", "--bounds", "10", "--radii", "3"])
        assert result.exit_code != 0
        assert "2 or 3 entries" in result.output

    def test_particle_too_large(self, runner) -> None:
        result = runner.invoke(cli, ["generate", "--bounds", "10,10", "--radii", "8"])
        assert result.exit_code != 0
        assert "does not fit" in result.output

    def test_weights_and_gaussian_are_exclusive(self, runner) -> None:
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "50,50",
                "--radii",
                "2,4",
                "--weights",
                "1,1",
                "--gaussian",
                "3,1",
            ],
        )
        assert result.exit_code != 0
        assert "not both" in result.output

    def test_weights_length_must_match(self, runner) -> None:
        result = runner.invoke(
            cli,
            ["generate", "--bounds", "50,50", "--radii", "2,4", "--weights", "1"],
        )
        assert result.exit_code != 0
        assert "does not match" in result.output

    def test_lognormal_needs_parameters(self, runner) -> None:
        result = runner.invoke(
            cli, ["generate", "--bounds", "50,50", "--dist", "lognormal"]
        )
        assert result.exit_code != 0
        assert "--gmean and --gstd" in result.output

    def test_unknown_output_suffix(self, runner, tmp_path) -> None:
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "50,50",
                "--radii",
                "4",
                "--seed",
                "0",
                "--out",
                str(tmp_path / "pack.txt"),
            ],
        )
        assert result.exit_code != 0
        assert ".csv or .npz" in result.output


class TestConfig:
    def test_toml_from_stdin(self, runner) -> None:
        result = runner.invoke(
            cli,
            ["generate", "--config", "-", "--no-report"],
            input="bounds = [50.0, 50.0]\nradii = [5.0]\nseed = 0\n",
        )
        assert result.exit_code == 0, result.output
        assert _csv_rows(result.output)[0] == "x,y,r"

    def test_yaml_file(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("bounds: [50.0, 50.0]\nradii: [5.0]\nseed: 0\n")
        result = runner.invoke(cli, ["generate", "--config", str(cfg), "--no-report"])
        assert result.exit_code == 0, result.output

    def test_json_file(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"bounds": [50.0, 50.0], "radii": [5.0], "seed": 0}))
        result = runner.invoke(cli, ["generate", "--config", str(cfg), "--no-report"])
        assert result.exit_code == 0, result.output

    def test_cli_overrides_config(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("bounds = [50.0, 50.0]\nradii = [5.0]\nseed = 0\n")
        out = tmp_path / "pack.npz"
        result = runner.invoke(
            cli,
            ["generate", "--config", str(cfg), "--seed", "123", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert Packing.load(out).seed == 123

    def test_profiles_merge_over_top_level_keys(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "seed = 3\nradii = [4.0]\n\n[profiles.small]\nbounds = [40.0, 40.0]\n"
            "\n[profiles.big]\nbounds = [90.0, 90.0]\n"
        )
        out = tmp_path / "pack.npz"
        result = runner.invoke(
            cli,
            ["generate", "--config", str(cfg), "--profile", "big", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        np.testing.assert_allclose(pack.bounds, [90.0, 90.0])
        assert pack.seed == 3

    def test_unknown_profile_lists_the_known_ones(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("[profiles.small]\nbounds = [40.0, 40.0]\nradii = [2.0]\n")
        result = runner.invoke(
            cli, ["generate", "--config", str(cfg), "--profile", "nope"]
        )
        assert result.exit_code != 0
        assert "small" in result.output

    def test_unsupported_config_suffix(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.ini"
        cfg.write_text("nope\n")
        result = runner.invoke(cli, ["generate", "--config", str(cfg)])
        assert result.exit_code != 0
        assert ".toml" in result.output


class TestBallContainer:
    def test_fills_a_ball(self, runner, tmp_path) -> None:
        out = tmp_path / "ball.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--shape",
                "ball",
                "--ball-radius",
                "30",
                "--radii",
                "2",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        assert pack.region.describe() == {
            "shape": "ball",
            "radius": 30.0,
            "dimensions": 3,
        }
        assert np.all(pack.region.contains(pack.points, pack.radii))

    def test_sphere_is_accepted_as_a_synonym(self, runner, tmp_path) -> None:
        out = tmp_path / "ball.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--shape",
                "sphere",
                "--ball-radius",
                "20",
                "--radii",
                "2",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert Packing.load(out).region.describe()["shape"] == "ball"

    def test_two_dimensional_ball(self, runner, tmp_path) -> None:
        out = tmp_path / "disc.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--shape",
                "ball",
                "--ball-radius",
                "25",
                "--dims",
                "2",
                "--radii",
                "2",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        assert pack.dimensions == 2
        assert pack.region.radius == pytest.approx(25.0)

    def test_ball_needs_a_radius(self, runner) -> None:
        result = runner.invoke(cli, ["generate", "--shape", "ball", "--radii", "2"])
        assert result.exit_code != 0
        assert "--ball-radius" in result.output

    def test_report_names_the_shape(self, runner) -> None:
        result = runner.invoke(
            cli,
            [
                "generate",
                "--shape",
                "ball",
                "--ball-radius",
                "20",
                "--radii",
                "2",
                "--seed",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ball, radius 20" in result.stderr

    def test_ball_from_config(self, runner, tmp_path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('shape = "ball"\nball_radius = 20.0\nradii = [2.0]\nseed = 0\n')
        out = tmp_path / "ball.npz"
        result = runner.invoke(
            cli, ["generate", "--config", str(cfg), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert Packing.load(out).region.radius == pytest.approx(20.0)


class TestCarpetRecipe:
    def test_a_thin_box_needs_no_special_command(self, runner, tmp_path) -> None:
        out = tmp_path / "carpet.npz"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "80,80,4.1",
                "--radii",
                "2",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        pack = Packing.load(out)
        np.testing.assert_allclose(pack.bounds, [80.0, 80.0, 4.1])
        assert np.all(pack.points[:, 2] <= 2.1 + 1e-9)

    def test_spacing_needs_the_depth_pre_inflated(self, runner, tmp_path) -> None:
        too_thin = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "80,80,4.1",
                "--radii",
                "2",
                "--spacing",
                "2",
                "--seed",
                "0",
            ],
        )
        assert too_thin.exit_code != 0
        assert "does not fit" in too_thin.output

        out = tmp_path / "carpet.npz"
        ok = runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "80,80,8.2",
                "--radii",
                "2",
                "--spacing",
                "2",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        assert ok.exit_code == 0, ok.output
        assert Packing.load(out).n > 0


class TestReportCommand:
    def test_text_report(self, runner, tmp_path) -> None:
        out = tmp_path / "pack.npz"
        runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "60,60",
                "--radii",
                "3",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        result = runner.invoke(cli, ["report", str(out)])
        assert result.exit_code == 0, result.output
        assert "fill fraction" in result.output

    def test_json_report(self, runner, tmp_path) -> None:
        out = tmp_path / "pack.npz"
        runner.invoke(
            cli,
            [
                "generate",
                "--bounds",
                "60,60",
                "--radii",
                "3",
                "--seed",
                "0",
                "--out",
                str(out),
            ],
        )
        result = runner.invoke(cli, ["report", str(out), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["dimensions"] == 2

    def test_missing_file(self, runner) -> None:
        assert runner.invoke(cli, ["report", "/nonexistent.npz"]).exit_code != 0


class TestRadiiCommand:
    def test_toml_output(self, runner) -> None:
        result = runner.invoke(
            cli, ["radii", "--rmin", "1", "--rmax", "3", "--rcount", "3"]
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "radii = [1, 2, 3]"

    def test_json_output_with_gaussian_weights(self, runner) -> None:
        result = runner.invoke(
            cli,
            [
                "radii",
                "--rmin",
                "1",
                "--rmax",
                "3",
                "--rcount",
                "3",
                "--gaussian",
                "2,0.5",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["radii"] == [1.0, 2.0, 3.0]
        assert sum(payload["weights"]) == pytest.approx(1.0)
        assert payload["weights"][1] == max(payload["weights"])

    def test_csv_output(self, runner) -> None:
        result = runner.invoke(
            cli,
            ["radii", "--rmin", "1", "--rmax", "2", "--rcount", "2", "--format", "csv"],
        )
        assert result.exit_code == 0, result.output
        assert _csv_rows(result.output) == ["r", "1", "2"]

    def test_gaussian_needs_two_numbers(self, runner) -> None:
        result = runner.invoke(
            cli, ["radii", "--rmin", "1", "--rmax", "2", "--gaussian", "2"]
        )
        assert result.exit_code != 0
        assert "mean,sigma" in result.output


def test_version(runner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "polydisc" in result.output
