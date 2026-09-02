"""Command-line interface for :mod:`polydisc`.

Every sampling option can come from the command line or from a
``.toml``/``.yaml``/``.json`` config file (optionally with named
profiles); explicit command-line values always win.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

from polydisc import __version__
from polydisc.diagnostics import report as report_fn
from polydisc.radii import RadiusSpec
from polydisc.regions import Ball, Box, Region
from polydisc.sampling import Packing, sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    """Send warnings (and with `verbose`, info) to stderr."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,
        )


def _load_config(config_path: str, config_format: str | None) -> dict[str, Any]:
    """Read a config file (or stdin) into a mapping.

    Parameters
    ----------
    config_path : str
        Path to a ``.toml``/``.yaml``/``.yml``/``.json`` file, or
        ``"-"`` to read stdin.
    config_format : str or None
        Format for stdin input; ignored for paths with a known suffix.

    Returns
    -------
    dict
        The parsed mapping, empty if the document is not a mapping.

    Raises
    ------
    click.BadParameter
        If the format or file suffix is not recognised.
    """

    def _parse(raw: bytes, fmt: str) -> dict[str, Any]:
        match fmt:
            case "toml":
                loaded: Any = tomllib.loads(raw.decode("utf-8"))
            case "yaml" | "yml":
                loaded = yaml.safe_load(raw)
            case "json":
                loaded = json.loads(raw.decode("utf-8"))
            case _:
                raise click.BadParameter(
                    "--config-format must be one of: toml, yaml, json"
                )
        return loaded if isinstance(loaded, dict) else {}

    if config_path == "-":
        return _parse(sys.stdin.buffer.read(), (config_format or "toml").lower())

    path = Path(config_path)
    match path.suffix.lower():
        case ".toml":
            fmt = "toml"
        case ".yaml" | ".yml":
            fmt = "yaml"
        case ".json":
            fmt = "json"
        case _:
            raise click.BadParameter("config file must be .toml, .yaml/.yml, or .json")
    return _parse(path.read_bytes(), fmt)


def _resolve_profile(config: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Merge a named profile over the config's top-level scalars.

    Parameters
    ----------
    config : dict
        The full parsed config.
    profile : str or None
        Name of a key inside the ``profiles`` table.  ``None`` returns
        `config` unchanged.

    Returns
    -------
    dict
        The effective settings.

    Raises
    ------
    click.BadParameter
        If there is no ``profiles`` table or no such profile.
    """
    if profile is None:
        return config

    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise click.BadParameter("config has no [profiles] table / profiles map")

    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        known = ", ".join(sorted(profiles)) or "(none)"
        raise click.BadParameter(f"unknown profile {profile!r}; available: {known}")

    base = {
        k: v for k, v in config.items() if k != "profiles" and not isinstance(v, dict)
    }
    return {**base, **selected}


def _pick(
    cfg: dict[str, Any] | None,
    key: str,
    cli_value: Any,
    *,
    coerce: str | None = None,
) -> Any:
    """Return the CLI value if given, else the config value.

    Parameters
    ----------
    cfg : dict or None
        Effective config mapping.
    key : str
        Config key to fall back to.
    cli_value : Any
        Value from the command line; ``None`` or ``""`` means unset.
    coerce : {None, 'int', 'float', 'bool', 'list_float'}, optional
        Conversion applied to a value taken from the config.

    Returns
    -------
    Any
        The resolved value, or ``None`` when neither source has one.

    Raises
    ------
    click.BadParameter
        If a config value cannot be coerced as requested.
    """
    if cli_value is not None and cli_value != "":
        return cli_value
    if cfg is None:
        return cli_value
    val = cfg.get(key)
    if val is None:
        return cli_value
    try:
        match coerce:
            case "int":
                return int(val)
            case "float":
                return float(val)
            case "bool":
                return bool(val)
            case "list_float":
                if not isinstance(val, (list, tuple)):
                    raise TypeError
                return np.asarray([float(x) for x in val], dtype=np.float64)
            case _:
                return val
    except (TypeError, ValueError) as exc:
        raise click.BadParameter(
            f"config key {key!r} has an unusable value {val!r}"
        ) from exc


def _parse_csv_floats(value: str | None) -> np.ndarray | None:
    """Parse ``"1,2,3"`` into a float array, or ``None`` if empty."""
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return None
    try:
        return np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise click.BadParameter(
            f"expected comma-separated numbers, got {value!r}"
        ) from exc


def _parse_pair(value: Any, *, option_name: str) -> tuple[float, float]:
    """Parse ``"a,b"`` or ``{"mean": a, "sigma": b}`` into two floats.

    Parameters
    ----------
    value : str or dict
        The raw option value.
    option_name : str
        Name used in error messages.

    Returns
    -------
    tuple of float
        The two parsed values.

    Raises
    ------
    click.BadParameter
        If the value has the wrong shape or is not numeric.
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise click.BadParameter(f"{option_name} expects mean,sigma")
        try:
            return float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise click.BadParameter(f"{option_name} expects two numbers") from exc

    if isinstance(value, dict):
        keys = ("mean", "sigma")
        if not all(k in value for k in keys):
            raise click.BadParameter(f"{option_name} expects keys: mean, sigma")
        return float(value["mean"]), float(value["sigma"])

    raise click.BadParameter(
        f"{option_name} expects mean,sigma or a {{mean, sigma}} object"
    )


def _gaussian_weights(radii: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Weight a discrete radius set by a Gaussian centred on `mean`."""
    if sigma <= 0:
        raise click.BadParameter("--gaussian sigma must be > 0")
    z = (radii - mean) / sigma
    w = np.exp(-0.5 * z * z)
    total = float(np.sum(w))
    if total == 0:
        raise click.ClickException(
            "gaussian weights are all zero; is the mean far outside the radius range?"
        )
    return w / total


# ---------------------------------------------------------------------
# radius spec resolution
# ---------------------------------------------------------------------


def _build_spec(
    cfg: dict[str, Any] | None,
    *,
    dist: str | None,
    radii: str | None,
    rmin: float | None,
    rmax: float | None,
    rcount: int | None,
    weights: str | None,
    gaussian: str | None,
    gmean: float | None,
    gstd: float | None,
    mean: float | None,
    std: float | None,
) -> RadiusSpec:
    """Turn CLI/config radius options into a :class:`RadiusSpec`.

    Returns
    -------
    RadiusSpec
        The resolved distribution.

    Raises
    ------
    click.BadParameter
        If required parameters for the chosen distribution are missing
        or mutually exclusive ones are combined.
    """
    dist = (_pick(cfg, "dist", dist) or "discrete").lower()
    rmin = _pick(cfg, "rmin", rmin, coerce="float")
    rmax = _pick(cfg, "rmax", rmax, coerce="float")

    if dist == "lognormal":
        gmean = _pick(cfg, "gmean", gmean, coerce="float")
        gstd = _pick(cfg, "gstd", gstd, coerce="float")
        if gmean is None or gstd is None:
            raise click.BadParameter("--dist lognormal needs --gmean and --gstd")
        return RadiusSpec.lognormal(gmean, gstd, rmin=rmin, rmax=rmax)

    if dist == "normal":
        mean = _pick(cfg, "mean", mean, coerce="float")
        std = _pick(cfg, "std", std, coerce="float")
        if mean is None or std is None:
            raise click.BadParameter("--dist normal needs --mean and --std")
        return RadiusSpec.normal(mean, std, rmin=rmin, rmax=rmax)

    if dist == "uniform":
        if rmin is None or rmax is None:
            raise click.BadParameter("--dist uniform needs --rmin and --rmax")
        return RadiusSpec.uniform(rmin, rmax)

    if dist != "discrete":
        raise click.BadParameter(
            f"unknown --dist {dist!r}; expected discrete, lognormal, normal or uniform"
        )

    values = _parse_csv_floats(radii)
    if values is None:
        values = _pick(cfg, "radii", None, coerce="list_float")
    if values is None:
        rcount = _pick(cfg, "rcount", rcount, coerce="int")
        if rmin is None or rmax is None or rcount is None:
            raise click.BadParameter(
                "discrete radii need --radii, or --rmin/--rmax/--rcount"
            )
        values = np.linspace(float(rmin), float(rmax), int(rcount))

    values = np.asarray(values, dtype=np.float64)

    if weights is not None and gaussian is not None:
        raise click.BadParameter("use either --weights or --gaussian, not both")

    weight_arr: np.ndarray | None = None
    if weights is not None:
        weight_arr = _parse_csv_floats(weights)
    elif gaussian is not None:
        w_mean, w_sigma = _parse_pair(gaussian, option_name="--gaussian")
        weight_arr = _gaussian_weights(values, w_mean, w_sigma)
    else:
        cfg_weights = _pick(cfg, "weights", None, coerce="list_float")
        cfg_gaussian = _pick(cfg, "gaussian", None)
        if cfg_weights is not None:
            weight_arr = np.asarray(cfg_weights, dtype=np.float64)
        elif cfg_gaussian is not None:
            w_mean, w_sigma = _parse_pair(cfg_gaussian, option_name="gaussian")
            weight_arr = _gaussian_weights(values, w_mean, w_sigma)

    if weight_arr is not None and weight_arr.shape != values.shape:
        raise click.BadParameter(
            f"weights length {weight_arr.shape[0]} does not match "
            f"{values.shape[0]} radii"
        )

    try:
        return RadiusSpec.discrete(values, weight_arr)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _build_region(
    cfg: dict[str, Any] | None,
    *,
    shape: str | None,
    bounds: str | None,
    ball_radius: float | None,
    dims: int | None,
) -> Region:
    """Turn CLI/config container options into a region.

    Returns
    -------
    Box or Ball
        The resolved container.

    Raises
    ------
    click.BadParameter
        If the shape is unknown or its required parameters are missing
        or malformed.
    """
    shape = (_pick(cfg, "shape", shape) or "box").lower()

    if shape in ("ball", "sphere"):
        radius = _pick(cfg, "ball_radius", ball_radius, coerce="float")
        if radius is None:
            raise click.BadParameter(f"--shape {shape} needs --ball-radius")
        dimensions = _pick(cfg, "dims", dims, coerce="int")
        try:
            return Ball(float(radius), 3 if dimensions is None else int(dimensions))
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc

    if shape != "box":
        raise click.BadParameter(f"unknown --shape {shape!r}; expected box or ball")

    value = _pick(cfg, "bounds", bounds, coerce="list_float")
    if value is None:
        raise click.BadParameter("--bounds is required (or set bounds in config)")
    edges = (
        np.asarray(value, dtype=np.float64)
        if isinstance(value, np.ndarray)
        else _parse_csv_floats(str(value))
    )
    if edges is None:
        raise click.BadParameter("--bounds must list 2 or 3 numbers")
    try:
        return Box(edges)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


# ---------------------------------------------------------------------
# output
# ---------------------------------------------------------------------


def _dump_csv(points: np.ndarray, radii: np.ndarray) -> str:
    """Render points and radii as CSV text with an ``x,y[,z],r`` header."""
    header = "x,y,r" if points.shape[1] == 2 else "x,y,z,r"
    buf = io.StringIO()
    buf.write(header + "\n")
    for p, r in zip(points, radii):
        buf.write(",".join([*(f"{v:.10g}" for v in p), f"{float(r):.10g}"]))
        buf.write("\n")
    return buf.getvalue()


def _emit(pack: Packing, out: str | None, show_report: bool) -> None:
    """Write a packing to `out` and optionally print diagnostics."""
    if show_report:
        click.echo(str(report_fn(pack)), err=True)

    if out is None or out == "-":
        click.echo(_dump_csv(pack.points, pack.radii), nl=False)
        return

    out_path = Path(out)
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_dump_csv(pack.points, pack.radii))
    elif suffix == ".npz":
        pack.save(out_path)
    else:
        raise click.BadParameter("--out must end in .csv or .npz")
    click.echo(f"Wrote {pack.n} particles to {out_path}", err=True)


# ---------------------------------------------------------------------
# shared options
# ---------------------------------------------------------------------


def _sampling_options[F: Callable[..., Any]](func: F) -> F:
    """Attach the sampling options for the generate command."""
    options = [
        click.option(
            "--config",
            default=None,
            help="Config file (.toml/.yaml/.json), or '-' for stdin",
        ),
        click.option(
            "--config-format",
            type=click.Choice(["toml", "yaml", "json"], case_sensitive=False),
            default="toml",
            show_default=True,
            help="Format for --config - (stdin)",
        ),
        click.option(
            "--profile",
            default=None,
            help="Profile name inside the config (profiles.<name>)",
        ),
        click.option(
            "--dist",
            type=click.Choice(
                ["discrete", "lognormal", "normal", "uniform"],
                case_sensitive=False,
            ),
            default=None,
            help="Radius distribution family  [default: discrete]",
        ),
        click.option("--radii", default=None, help="Comma-separated allowed radii"),
        click.option("--rmin", type=float, default=None, help="Smallest radius"),
        click.option("--rmax", type=float, default=None, help="Largest radius"),
        click.option(
            "--rcount",
            type=click.IntRange(1, 10_000),
            default=None,
            help="Number of discrete radii between --rmin and --rmax",
        ),
        click.option(
            "--weights", default=None, help="Comma-separated discrete weights"
        ),
        click.option(
            "--gaussian",
            default=None,
            help="Gaussian weights over the discrete radii, as mean,sigma",
        ),
        click.option(
            "--gmean", type=float, default=None, help="Lognormal geometric mean"
        ),
        click.option(
            "--gstd", type=float, default=None, help="Lognormal geometric std (>=1)"
        ),
        click.option("--mean", type=float, default=None, help="Normal mean radius"),
        click.option("--std", type=float, default=None, help="Normal std of radius"),
        click.option(
            "--spacing",
            type=float,
            default=None,
            help="Neighbour clearance in mean radii; 2 fits one average "
            "particle between neighbours  [default: 0]",
        ),
        click.option(
            "--fit-bounds/--no-fit-bounds",
            default=None,
            help="Keep the requested extent when spacing is applied "
            "(default), or let the domain grow by the spacing factor",
        ),
        click.option(
            "--repetitions",
            type=click.IntRange(1, 10_000),
            default=None,
            help="Candidates per active point  [default: 30]",
        ),
        click.option(
            "--max-points",
            type=click.IntRange(1),
            default=None,
            help="Stop after this many particles",
        ),
        click.option(
            "--seed",
            type=int,
            default=None,
            help="Random seed; omit for a fresh random packing",
        ),
        click.option(
            "--out",
            default=None,
            help="Output path (.csv or .npz); '-' or omitted prints CSV",
        ),
        click.option(
            "--report/--no-report",
            default=True,
            show_default=True,
            help="Print realised diagnostics to stderr",
        ),
        click.option("-v", "--verbose", is_flag=True, help="Log info messages"),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _resolve_common(
    cfg: dict[str, Any] | None,
    spacing: float | None,
    fit_bounds: bool | None,
    repetitions: int | None,
    max_points: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Resolve the non-radius sampling options from CLI plus config."""
    spacing = _pick(cfg, "spacing", spacing, coerce="float")
    fit_bounds = _pick(cfg, "fit_bounds", fit_bounds, coerce="bool")
    repetitions = _pick(cfg, "repetitions", repetitions, coerce="int")
    max_points = _pick(cfg, "max_points", max_points, coerce="int")
    seed = _pick(cfg, "seed", seed, coerce="int")
    return {
        "spacing": 0.0 if spacing is None else float(spacing),
        "fit_bounds": True if fit_bounds is None else bool(fit_bounds),
        "repetitions": 30 if repetitions is None else int(repetitions),
        "max_points": max_points,
        "seed": seed,
    }


# ---------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="polydisc")
def cli() -> None:
    """Polydisperse Poisson-disc sampling for discs and spheres."""


@cli.command("generate")
@click.option(
    "--shape",
    type=click.Choice(["box", "ball", "sphere"], case_sensitive=False),
    default=None,
    help="Container to fill  [default: box]",
)
@click.option(
    "--bounds",
    default=None,
    help="Box edges, e.g. 800,600 or 200,200,50. A flat carpet is a box "
    "whose last edge is about one particle diameter",
)
@click.option(
    "--ball-radius",
    type=float,
    default=None,
    help="Radius of the container, for --shape ball",
)
@click.option(
    "--dims",
    type=click.IntRange(2, 3),
    default=None,
    help="Dimension of a ball container  [default: 3]",
)
@_sampling_options
def generate(
    shape: str | None,
    bounds: str | None,
    ball_radius: float | None,
    dims: int | None,
    config: str | None,
    **kwargs: Any,
) -> None:
    """Fill a box or a ball with a packing.

    \b
    Examples:
      polydisc generate --bounds 500,500 --radii 2,4 --seed 0
      polydisc generate --shape ball --ball-radius 100 --radii 2,4
      polydisc generate --bounds 500,500,4.1 --dist lognormal \\
          --gmean 1 --gstd 1.3 --spacing 2 --out carpet.npz
    """
    _configure_logging(kwargs.pop("verbose", False))
    cfg = _config_from(config, kwargs.pop("config_format"), kwargs.pop("profile"))

    region = _build_region(
        cfg, shape=shape, bounds=bounds, ball_radius=ball_radius, dims=dims
    )
    spec = _build_spec(cfg, **_spec_kwargs(kwargs))
    common = _resolve_common(
        cfg,
        kwargs.pop("spacing"),
        kwargs.pop("fit_bounds"),
        kwargs.pop("repetitions"),
        kwargs.pop("max_points"),
        kwargs.pop("seed"),
    )

    try:
        pack = sample(region, spec, **common)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit(pack, kwargs.pop("out"), kwargs.pop("report"))


@cli.command("report")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the report as JSON instead of a text summary",
)
def report_cmd(path: str, as_json: bool) -> None:
    """Print diagnostics for a packing saved as .npz."""
    pack = Packing.load(path)
    rep = report_fn(pack)
    if as_json:
        click.echo(json.dumps(rep.to_dict(), indent=2))
    else:
        click.echo(str(rep))


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
    help="Output format; TOML/JSON paste straight into a config",
)
def radii_cmd(
    rmin: float,
    rmax: float,
    rcount: int,
    gaussian: str | None,
    output_format: str,
) -> None:
    """Print a discrete radius set, optionally Gaussian-weighted."""
    values = np.linspace(float(rmin), float(rmax), int(rcount), dtype=np.float64)

    weights: np.ndarray | None = None
    if gaussian is not None:
        mean, sigma = _parse_pair(gaussian, option_name="--gaussian")
        weights = _gaussian_weights(values, mean, sigma)

    match output_format.lower():
        case "json":
            out: dict[str, Any] = {"radii": values.tolist()}
            if weights is not None:
                out["weights"] = weights.tolist()
            click.echo(json.dumps(out, indent=2))
        case "toml":
            click.echo(
                "radii = [" + ", ".join(f"{float(r):.10g}" for r in values) + "]"
            )
            if weights is not None:
                click.echo(
                    "weights = [" + ", ".join(f"{float(w):.10g}" for w in weights) + "]"
                )
        case _:
            if weights is None:
                click.echo("r")
                for r in values:
                    click.echo(f"{float(r):.10g}")
            else:
                click.echo("r,weight")
                for r, w in zip(values, weights):
                    click.echo(f"{float(r):.10g},{float(w):.10g}")


# ---------------------------------------------------------------------
# helpers used by the commands
# ---------------------------------------------------------------------

_SPEC_KEYS = (
    "dist",
    "radii",
    "rmin",
    "rmax",
    "rcount",
    "weights",
    "gaussian",
    "gmean",
    "gstd",
    "mean",
    "std",
)


def _spec_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop the radius-related options out of a command's kwargs."""
    return {key: kwargs.pop(key) for key in _SPEC_KEYS}


def _config_from(
    config: str | None, config_format: str | None, profile: str | None
) -> dict[str, Any] | None:
    """Load and profile-resolve a config, or return ``None``."""
    if config is None:
        return None
    return _resolve_profile(_load_config(config, config_format), profile)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    return cli.main(args=argv, standalone_mode=True)  # type: ignore[no-any-return]
