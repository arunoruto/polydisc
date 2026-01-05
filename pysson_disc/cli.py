from __future__ import annotations

import io
import json
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import yaml

from pysson_disc.poisson_disc import poisson_disc_samples

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # Click CLIs often appear to "do nothing" if they only log.
    # Default to INFO so users get a clear "Wrote ..." message.
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_csv_floats(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def _weights_from_gaussian(radii: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise click.BadParameter("sigma must be > 0")
    z = (radii - mean) / sigma
    w = np.exp(-0.5 * z * z)
    s = float(np.sum(w))
    if s == 0:
        raise click.ClickException("gaussian weights sum to 0")
    return w / s


def _parse_gaussian(value: Any, *, option_name: str) -> tuple[float, float]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise click.BadParameter(f"{option_name} expects mean,sigma")
        return float(parts[0]), float(parts[1])

    if isinstance(value, dict):
        if "mean" not in value or "sigma" not in value:
            raise click.BadParameter(f"{option_name} expects keys: mean, sigma")
        return float(value["mean"]), float(value["sigma"])

    raise click.BadParameter(
        f"{option_name} expects mean,sigma or a {{mean, sigma}} object"
    )


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return None


def _load_config(config_path: str, config_format: str | None) -> dict[str, Any]:
    def _load_from_bytes(raw: bytes, fmt: str) -> dict[str, Any]:
        match fmt:
            case "toml":
                loaded = tomllib.loads(raw.decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}
            case "yaml" | "yml":
                loaded = yaml.safe_load(raw)
                return loaded if isinstance(loaded, dict) else {}
            case "json":
                loaded = json.loads(raw.decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}
            case _:
                raise click.BadParameter(
                    "--config-format must be one of: toml, yaml, json"
                )

    if config_path == "-":
        raw = sys.stdin.buffer.read()
        fmt = (config_format or "toml").lower()
        return _load_from_bytes(raw, fmt)

    path = Path(config_path)
    raw_bytes = path.read_bytes()
    suffix = path.suffix.lower()

    fmt: str
    match suffix:
        case ".toml":
            fmt = "toml"
        case ".yaml" | ".yml":
            fmt = "yaml"
        case ".json":
            fmt = "json"
        case _:
            raise click.BadParameter("Config file must be .toml, .yaml/.yml, or .json")

    return _load_from_bytes(raw_bytes, fmt)


def _resolve_profile(config: dict[str, Any], profile: str | None) -> dict[str, Any]:
    if profile is None:
        return config

    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise click.BadParameter("config has no [profiles] table / profiles map")

    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        raise click.BadParameter(f"unknown profile: {profile}")

    base = {
        k: v for k, v in config.items() if k != "profiles" and not isinstance(v, dict)
    }
    merged = {**base, **selected}
    return merged


def _get_config_value(cfg: dict[str, Any], key: str) -> Any:
    return cfg.get(key)


def _maybe_from_config(
    cfg: dict[str, Any] | None, key: str, cli_value: Any, *, coerce: str | None = None
) -> Any:
    if cli_value is not None and cli_value != "":
        return cli_value
    if cfg is None:
        return cli_value
    val = cfg.get(key)
    if val is None:
        return cli_value
    if coerce == "int":
        return int(val)
    if coerce == "float":
        return float(val)
    if coerce == "list_float":
        lst = _as_list(val)
        if lst is None:
            raise click.BadParameter(f"config key '{key}' must be a list")
        return np.asarray([float(x) for x in lst], dtype=np.float64)
    if coerce == "list_str":
        lst = _as_list(val)
        if lst is None:
            raise click.BadParameter(f"config key '{key}' must be a list")
        return [str(x) for x in lst]
    return val


def _dump_csv(points: np.ndarray, radii: np.ndarray) -> str:
    if points.shape[1] == 2:
        header = "x,y,r"
    elif points.shape[1] == 3:
        header = "x,y,z,r"
    else:
        raise click.ClickException("CSV output supports only 2D or 3D")

    buf = io.StringIO()
    buf.write(header + "\n")
    for p, r in zip(points, radii):
        row = ",".join([*(f"{v:.10g}" for v in p), f"{float(r):.10g}"])
        buf.write(row + "\n")
    return buf.getvalue()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Generate Poisson-disc samples in 2D/3D boxes."""


@cli.command("generate")
@click.option(
    "--config",
    default=None,
    help="Config file (.toml/.yaml/.json) or '-' for stdin (TOML by default)",
)
@click.option(
    "--config-format",
    type=click.Choice(["toml", "yaml", "json"], case_sensitive=False),
    default="toml",
    show_default=True,
    help="Format for --config - (stdin). Ignored for file paths with known suffix",
)
@click.option(
    "--profile", default=None, help="Profile name inside config (profiles.<name>)"
)
@click.option("--dims", type=click.IntRange(2, 3), default=None)
@click.option(
    "--bounds",
    default=None,
    help="Comma-separated bounds, e.g. 800,600 or 200,200,200",
)
@click.option("--rmin", type=float, default=None)
@click.option("--rmax", type=float, default=None)
@click.option("--rcount", type=click.IntRange(1, 10_000), default=None)
@click.option(
    "--weights",
    default=None,
    help="Comma-separated weights matching allowed radii count",
)
@click.option(
    "--gaussian",
    default=None,
    help="Gaussian weights as mean,sigma (over the allowed radii)",
)
@click.option("--seed", type=int, default=None)
@click.option("--repetitions", type=click.IntRange(1, 10_000), default=None)
@click.option(
    "--out",
    type=str,
    default=None,
    help="Output filepath (.csv or .npz). Use '-' or omit to print CSV data to stdout",
)
def generate(
    config: str | None,
    config_format: str | None,
    profile: str | None,
    dims: int | None,
    bounds: str | None,
    rmin: float | None,
    rmax: float | None,
    rcount: int | None,
    weights: str | None,
    gaussian: str | None,
    seed: int | None,
    repetitions: int | None,
    out: str | None,
) -> None:
    """Generate samples and write them to disk."""

    _configure_logging()

    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(".numba_cache").resolve()))

    cfg: dict[str, Any] | None = None
    if config is not None:
        loaded = _load_config(config, config_format=config_format)
        cfg = _resolve_profile(loaded, profile)

    dims = _maybe_from_config(cfg, "dims", dims, coerce="int")
    if dims is None:
        dims = 2

    bounds_val = _maybe_from_config(cfg, "bounds", bounds, coerce="list_float")
    if bounds_val is None:
        raise click.BadParameter("--bounds is required (or set bounds in config)")

    if isinstance(bounds_val, np.ndarray):
        bounds_arr = np.asarray(bounds_val, dtype=np.float64)
    else:
        bounds_arr = _parse_csv_floats(str(bounds_val))
        if bounds_arr is None:
            raise click.BadParameter("bounds must be a comma-separated list of numbers")

    if bounds_arr.shape[0] != int(dims):
        raise click.BadParameter(f"bounds must have {dims} values")

    repetitions = _maybe_from_config(cfg, "repetitions", repetitions, coerce="int")
    if repetitions is None:
        repetitions = 30

    seed = _maybe_from_config(cfg, "seed", seed, coerce="int")
    if seed is None:
        seed = 0
    np.random.seed(int(seed))

    radii_list = _maybe_from_config(cfg, "radii", None, coerce="list_float")
    if radii_list is not None:
        allowed_radii = np.asarray(radii_list, dtype=np.float64)
        if allowed_radii.ndim != 1 or allowed_radii.size == 0:
            raise click.BadParameter("radii must be a non-empty list")
        if np.any(allowed_radii <= 0):
            raise click.BadParameter("radii must be > 0")
    else:
        rmin = _maybe_from_config(cfg, "rmin", rmin, coerce="float")
        rmax = _maybe_from_config(cfg, "rmax", rmax, coerce="float")
        rcount = _maybe_from_config(cfg, "rcount", rcount, coerce="int")
        if rmin is None or rmax is None or rcount is None:
            raise click.BadParameter(
                "Must specify either 'radii' in config, or --rmin/--rmax/--rcount (or config equivalents)"
            )
        allowed_radii = np.linspace(
            float(rmin), float(rmax), int(rcount), dtype=np.float64
        )

    # weights + gaussian can come from config and/or CLI; CLI overrides.

    weights_arr: Optional[np.ndarray] = None

    # CLI wins over config: if the user passes --weights/--gaussian,
    # ignore weights/gaussian from config.
    weights_cfg = None
    if weights is None and gaussian is None:
        weights_cfg = _maybe_from_config(cfg, "weights", None, coerce="list_float")
    if weights_cfg is not None:
        weights_arr = np.asarray(weights_cfg, dtype=np.float64)
        if weights_arr.shape[0] != allowed_radii.shape[0]:
            raise click.BadParameter("weights length must match allowed radii count")
        if np.any(weights_arr < 0):
            raise click.BadParameter("weights must be non-negative")
        s = float(np.sum(weights_arr))
        if s <= 0:
            raise click.BadParameter("weights must sum to > 0")
        weights_arr = weights_arr / s

    if weights is not None and gaussian is not None:
        raise click.BadParameter("use either --weights or --gaussian, not both")

    if weights is not None:
        weights_arr = _parse_csv_floats(weights)
        if weights_arr is None or weights_arr.shape[0] != allowed_radii.shape[0]:
            raise click.BadParameter("weights length must match allowed radii count")
        if np.any(weights_arr < 0):
            raise click.BadParameter("weights must be non-negative")
        s = float(np.sum(weights_arr))
        if s <= 0:
            raise click.BadParameter("weights must sum to > 0")
        weights_arr = weights_arr / s

    gaussian_cfg = None
    if gaussian is None and weights is None:
        gaussian_cfg = _maybe_from_config(cfg, "gaussian", None)
    gaussian = gaussian if gaussian is not None else gaussian_cfg

    if gaussian is not None:
        mean, sigma = _parse_gaussian(gaussian, option_name="--gaussian")
        weights_arr = _weights_from_gaussian(allowed_radii, mean=mean, sigma=sigma)

    points, radii_out = poisson_disc_samples(
        bounds_arr, allowed_radii, repetitions=int(repetitions), weights=weights_arr
    )

    if out is None or out == "-":
        click.echo(_dump_csv(points, radii_out), nl=False)
        return

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        out_path.write_text(_dump_csv(points, radii_out))
        click.echo(f"Wrote {len(points)} samples to {out_path}")
        return

    if suffix == ".npz":
        np.savez_compressed(
            out_path,
            points=points,
            radii=radii_out,
            bounds=bounds_arr,
            allowed_radii=allowed_radii,
        )
        click.echo(f"Wrote {len(points)} samples to {out_path}")
        return

    raise click.BadParameter("--out must end in .csv or .npz")


@cli.command("radii")
@click.option("--rmin", type=float, required=True)
@click.option("--rmax", type=float, required=True)
@click.option("--rcount", type=click.IntRange(1, 10_000), default=5, show_default=True)
@click.option("--gaussian", default=None, help="Gaussian weights as mean,sigma")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["toml", "csv", "json"], case_sensitive=False),
    default="toml",
    show_default=True,
    help="Output format. Use TOML/JSON for piping into configs; CSV is export-only.",
)
def radii_cmd(
    rmin: float,
    rmax: float,
    rcount: int,
    gaussian: str | None,
    output_format: str,
) -> None:
    """Print allowed radii (+ optional gaussian weights).

    Note: configs only support TOML/YAML/JSON. CSV output is intended for export/humans.
    """
    allowed_radii = np.linspace(float(rmin), float(rmax), int(rcount), dtype=np.float64)

    weights: np.ndarray | None = None
    if gaussian is not None:
        mean, sigma = _parse_gaussian(gaussian, option_name="--gaussian")
        weights = _weights_from_gaussian(allowed_radii, mean=mean, sigma=sigma)

    fmt = output_format.lower()

    if fmt == "json":
        out: dict[str, object] = {"radii": allowed_radii.tolist()}
        if weights is not None:
            out["weights"] = weights.tolist()
        click.echo(json.dumps(out, indent=2))
        return

    if fmt == "toml":
        radii_toml = ", ".join(f"{float(r):.10g}" for r in allowed_radii)
        click.echo(f"radii = [{radii_toml}]")
        if weights is not None:
            weights_toml = ", ".join(f"{float(w):.10g}" for w in weights)
            click.echo(f"weights = [{weights_toml}]")
        return

    # CSV default (easy piping; minimal whitespace)
    if weights is None:
        click.echo("r")
        for r in allowed_radii:
            click.echo(f"{float(r):.10g}")
        return

    click.echo("r,weight")
    for r, w in zip(allowed_radii, weights):
        click.echo(f"{float(r):.10g},{float(w):.10g}")
