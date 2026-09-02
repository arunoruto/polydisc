"""Polydisc dashboard: generate packings, or inspect ones you upload.

Run locally with ``streamlit run dashboard/app.py``, or deploy the
folder to Streamlit Community Cloud -- see ``dashboard/README.md``.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# `streamlit run` puts the entrypoint's directory on sys.path, but
# nothing else does -- not AppTest, not `python dashboard/app.py`. Do it
# explicitly so the sibling modules import the same way everywhere.
#
# The second entry covers Community Cloud, which installs
# dashboard/requirements.txt but not the repo itself: fall back to the
# src/ layout when polydisc is not already importable.
for _path in (_HERE, _HERE.parent / "src"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import streamlit as st
import views
from loaders import LoadError, load_packing

from polydisc import __version__
from polydisc.diagnostics import report
from polydisc.radii import RadiusSpec
from polydisc.regions import Ball, Box, Region
from polydisc.sampling import Packing, sample
from polydisc.spacing import spacing_scale

#: Where a generated or uploaded packing lives between reruns, so
#: switching tabs never resamples.
_STATE_KEY = "packing"
#: Whether the current packing's spec was requested or inferred.
_INFERRED_KEY = "spec_inferred"


# ---------------------------------------------------------------------
# sidebar controls
# ---------------------------------------------------------------------


def container_controls() -> tuple[Region, str]:
    """Render the container controls and return the chosen region.

    Returns
    -------
    region : Box or Ball
        The container to fill.
    shape : str
        The label the user picked, so callers can tell a carpet preset
        from a plain 3D box.
    """
    shape = st.selectbox(
        "container",
        ["2D box", "3D box", "2D ball", "3D ball", "carpet (thin box)"],
        help="A carpet is not a separate shape: it is a box whose depth "
        "is about one particle diameter.",
    )

    if shape.endswith("ball"):
        radius = st.number_input("container radius", 1.0, value=100.0, step=10.0)
        return Ball(radius, 2 if shape.startswith("2D") else 3), shape

    if shape == "2D box":
        width = st.number_input("width", 1.0, value=300.0, step=10.0)
        height = st.number_input("height", 1.0, value=200.0, step=10.0)
        return Box([width, height]), shape

    width = st.number_input("x extent", 1.0, value=120.0, step=10.0)
    height = st.number_input("y extent", 1.0, value=120.0, step=10.0)
    if shape == "3D box":
        depth = st.number_input("z extent", 1.0, value=120.0, step=10.0)
        return Box([width, height, depth]), shape

    return Box([width, height, 1.0]), shape  # depth filled in later


def radius_controls() -> RadiusSpec:
    """Render the size-distribution controls and return the spec.

    Returns
    -------
    RadiusSpec
        The requested radius distribution.

    Raises
    ------
    streamlit.errors.Error
        Never directly; invalid combinations are reported in the UI and
        a safe fallback spec is returned.
    """
    family = st.selectbox(
        "size distribution",
        ["lognormal", "discrete", "normal", "uniform"],
        help="Continuous families are truncated and sampled inside the Numba kernel.",
    )

    if family == "lognormal":
        gmean = st.number_input("geometric mean radius", 0.001, value=1.0, step=0.1)
        gstd = st.number_input(
            "geometric std (>= 1)",
            1.0,
            value=1.3,
            step=0.05,
            help="1 is monodisperse. Truncated at +/- 2 GSD, matching "
            "FracVAL, so the realised spread is about gstd ** 0.88.",
        )
        return RadiusSpec.lognormal(float(gmean), float(gstd))

    if family == "normal":
        mean = st.number_input("mean radius", 0.001, value=1.0, step=0.1)
        std = st.number_input("std of radius", 0.0, value=0.2, step=0.05)
        return RadiusSpec.normal(float(mean), float(std))

    rmin = st.number_input("min radius", 0.001, value=1.0, step=0.1)
    rmax = st.number_input("max radius", 0.001, value=2.0, step=0.1)
    if rmax < rmin:
        st.warning("max radius is below min radius; swapping them.")
        rmin, rmax = rmax, rmin

    if family == "uniform":
        return RadiusSpec.uniform(float(rmin), float(rmax))

    count = st.slider("number of allowed radii", 1, 20, 3)
    values = np.linspace(float(rmin), float(rmax), int(count))
    if st.checkbox("gaussian weights", value=False):
        centre = st.number_input("weight centre", value=float(np.mean(values)))
        sigma = st.number_input("weight sigma", 0.001, value=0.5, step=0.1)
        z = (values - float(centre)) / float(sigma)
        return RadiusSpec.discrete(values, np.exp(-0.5 * z * z))
    return RadiusSpec.discrete(values)


def generate_panel() -> None:
    """Render the generation controls and sample on request."""
    region, shape = container_controls()
    spec = radius_controls()

    st.divider()
    spacing = st.slider(
        "spacing (mean radii between neighbours)",
        0.0,
        6.0,
        0.0,
        0.25,
        help="2 leaves room for one average particle between neighbouring "
        "surfaces. Positions are dilated; radii are untouched.",
    )
    fit_bounds = st.checkbox(
        "keep the requested size",
        value=True,
        help="Off: the container grows by the spacing factor instead, "
        "keeping the particle count (PyFracVAL's behaviour).",
    )

    if shape == "carpet (thin box)":
        # The pre-shrink hits every axis, so a monolayer's depth has to
        # be inflated by the same factor to survive it.
        inflate = spacing_scale(spacing) if fit_bounds else 1.0
        depth = 2.05 * spec.rmax * inflate
        region = Box([region.bounds[0], region.bounds[1], depth])
        st.caption(f"depth {depth:.4g}, the thinnest slab these radii allow")

    repetitions = st.slider(
        "candidates per active point (k)",
        1,
        60,
        30,
        help="Higher packs denser and costs more.",
    )
    seed = int(st.number_input("seed", 0, 2**31 - 2, 0, 1))

    if st.button("Generate", type="primary", width="stretch"):
        try:
            with st.spinner("Sampling..."):
                pack = sample(
                    region,
                    spec,
                    repetitions=repetitions,
                    seed=seed,
                    spacing=spacing,
                    fit_bounds=fit_bounds,
                )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state[_STATE_KEY] = pack
        st.session_state[_INFERRED_KEY] = False


def upload_panel() -> None:
    """Render the file uploader and load on request."""
    st.caption(
        "Upload a `.npz` or `.csv` written by `polydisc generate`, or any "
        "file with the same columns."
    )
    uploaded = st.file_uploader(
        "packing file", type=["npz", "csv"], label_visibility="collapsed"
    )
    if uploaded is None:
        return

    try:
        pack = load_packing(uploaded.name, uploaded.getvalue())
    except LoadError as exc:
        st.error(str(exc))
        return

    st.session_state[_STATE_KEY] = pack
    # A .npz from this tool carries its spec; anything else was guessed
    # from the data, and comparing it against itself proves nothing.
    st.session_state[_INFERRED_KEY] = not uploaded.name.lower().endswith(".npz")
    st.success(f"Loaded {pack.n} particles from {uploaded.name}")


# ---------------------------------------------------------------------
# result tabs
# ---------------------------------------------------------------------


def view_tab(pack: Packing) -> None:
    """Render the geometry view for a packing."""
    if pack.dimensions == 2:
        figure, reduced = views.plot_2d(pack)
        if reduced:
            st.info("Showing a random subset; the full packing is in the download.")
        st.pyplot(figure, width="stretch")
        return

    mode = st.radio(
        "view",
        ["3D point cloud", "cross-section"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "3D point cloud":
        figure, reduced = views.plot_3d(pack)
        if reduced:
            st.info("Showing a random subset; the full packing is in the download.")
        st.caption(
            "Marker sizes track relative radii but are in screen pixels, "
            "so they do not rescale on zoom. Use the cross-section for "
            "true geometric scale."
        )
        st.plotly_chart(figure, width="stretch")
        return

    axis_name = st.radio(
        "slice normal", ["z", "y", "x"], horizontal=True, key="slice_axis"
    )
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    extent = float(pack.bounds[axis])
    default_thickness = min(extent, 4.0 * float(pack.radii.mean()))

    columns = st.columns(2)
    centre = columns[0].slider(
        f"{axis_name} position", 0.0, extent, extent / 2.0, extent / 200.0
    )
    thickness = columns[1].slider(
        "slab thickness",
        extent / 100.0,
        extent,
        float(default_thickness),
        extent / 100.0,
    )

    figure, shown = views.plot_slice(pack, axis, centre, thickness)
    st.caption(
        f"{shown} particles intersect the slab, drawn at the radius of "
        "their cross-section."
    )
    st.pyplot(figure, width="stretch")


def distribution_tab(pack: Packing, inferred: bool) -> None:
    """Render the realised size distribution and its biases."""
    rep = report(pack)
    st.plotly_chart(
        views.plot_radius_histogram(pack, show_requested=not inferred),
        width="stretch",
    )

    if inferred:
        st.info(
            "This file carries no requested distribution, so there is "
            "nothing to compare against; the curve is omitted."
        )
        return

    radii = rep.radii
    wall = rep.boundary_bias.get("bias") or 0.0
    columns = st.columns(3)
    columns[0].metric(
        "mean radius",
        f"{radii['mean']:.4g}",
        f"{radii['mean_bias'] * 100:+.1f}% vs requested",
        delta_color="off",
    )
    columns[1].metric("walls alone", f"{wall * 100:+.1f}%")
    columns[2].metric("CDF distance (KS)", f"{radii['ks']:.3f}")

    st.markdown(
        "Two effects move the realised sizes away from the request. "
        "**Boundary exclusion** is geometric: a particle of radius `r` "
        "only fits in the sub-region its own size leaves free, so large "
        "particles have fewer places to go. **Crowding** is the rest: "
        "candidate radii are drawn per placement attempt, so a small "
        "candidate survives a tight neighbourhood more often."
    )
    if abs(wall) > 0.05:
        st.warning(
            f"The container is choosing the sizes: it shifts the mean "
            f"radius {wall * 100:+.1f}% before any crowding. Enlarge it "
            "or narrow the radius range."
        )


def neighbours_tab(pack: Packing) -> None:
    """Render nearest-neighbour clearance statistics."""
    rep = report(pack)
    gaps = rep.gaps

    columns = st.columns(4)
    columns[0].metric("min gap", f"{gaps['min']:.4g}")
    columns[1].metric("median gap", f"{gaps['median']:.4g}")
    columns[2].metric("median / mean radius", f"{gaps['median_over_r']:.3f}")
    columns[3].metric("overlaps", gaps["overlaps"])

    if gaps["overlaps"]:
        st.error(
            f"{gaps['overlaps']} particles overlap a neighbour "
            f"(worst {gaps['worst_overlap']:.3g}). This should never "
            "happen for a generated packing."
        )
    else:
        st.success("No overlaps: every particle clears its nearest neighbour.")

    st.plotly_chart(views.plot_gap_histogram(pack), width="stretch")

    if pack.spacing > 0:
        st.caption(
            f"Spacing {pack.spacing:g} dilates positions by "
            f"{pack.scale:g}x, which opens `spacing * r_mean` between two "
            "touching average particles. Real neighbours are neither "
            "touching nor average, so the distribution sits above the "
            "marked line."
        )


def report_tab(pack: Packing) -> None:
    """Render the full text and JSON report."""
    rep = report(pack)
    st.code(str(rep), language="text")
    with st.expander("As JSON"):
        st.json(_jsonify(rep.to_dict()))
    with st.expander("Run settings"):
        st.json(_jsonify(pack.metadata()))


def download_tab(pack: Packing) -> None:
    """Offer the packing as CSV and NPZ."""
    header = "x,y,r" if pack.dimensions == 2 else "x,y,z,r"
    csv_buffer = io.StringIO()
    csv_buffer.write(header + "\n")
    for point, radius in zip(pack.points, pack.radii):
        csv_buffer.write(
            ",".join([*(f"{v:.10g}" for v in point), f"{float(radius):.10g}"]) + "\n"
        )

    npz_buffer = io.BytesIO()
    np.savez_compressed(
        npz_buffer,
        points=pack.points,
        radii=pack.radii,
        bounds=pack.bounds,
        metadata=np.asarray(json.dumps(pack.metadata())),
    )
    npz_buffer.seek(0)

    columns = st.columns(2)
    columns[0].download_button(
        "Download NPZ",
        data=npz_buffer,
        file_name=f"polydisc_{pack.dimensions}d.npz",
        mime="application/octet-stream",
        width="stretch",
        help="Carries the full run settings, so it reloads here or with "
        "polydisc.Packing.load with everything intact.",
    )
    columns[1].download_button(
        "Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"polydisc_{pack.dimensions}d.csv",
        mime="text/csv",
        width="stretch",
        help="Coordinates and radii only.",
    )

    st.code(
        f"import polydisc\n"
        f"pack = polydisc.Packing.load('polydisc_{pack.dimensions}d.npz')\n"
        f"print(polydisc.report(pack))",
        language="python",
    )


def _jsonify(value: object) -> object:
    """Convert NumPy scalars and non-finite floats for ``st.json``."""
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


# ---------------------------------------------------------------------
# page
# ---------------------------------------------------------------------


def summary_metrics(pack: Packing) -> None:
    """Render the headline numbers for a packing."""
    rep = report(pack)
    columns = st.columns(5)
    columns[0].metric("particles", f"{pack.n:,}")
    columns[1].metric("container", pack.region.label.split(",")[0])
    columns[2].metric("fill fraction", f"{pack.fill_fraction:.3f}")
    if pack.dimensions == 3:
        columns[3].metric("projected area", f"{rep.projected_area_fraction:.3f}")
    else:
        columns[3].metric("mean radius", f"{rep.radii['mean']:.4g}")
    columns[4].metric("saturated", "yes" if pack.saturated else "no")
    st.caption(pack.region.label)


def main() -> None:
    """Run the dashboard."""
    st.set_page_config(
        page_title="polydisc",
        page_icon="○",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("polydisc")
    st.caption(
        f"Polydisperse Poisson-disc sampling for discs and spheres · v{__version__}"
    )

    with st.sidebar:
        st.subheader("Source")
        source = st.radio(
            "source",
            ["Generate", "Upload a file"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.divider()
        if source == "Generate":
            generate_panel()
        else:
            upload_panel()

    pack: Packing | None = st.session_state.get(_STATE_KEY)
    if pack is None:
        st.info(
            "Set up a container and a size distribution in the sidebar and "
            "press **Generate**, or switch to **Upload a file** to inspect "
            "a packing you already have."
        )
        with st.expander("What this does", expanded=True):
            st.markdown(
                "Every particle gets its own radius drawn from a "
                "distribution you choose, no two overlap, and none pokes "
                "outside the container. The dashboard also reports what "
                "was **actually** produced: the realised size "
                "distribution against the requested one, and the "
                "nearest-neighbour clearances against the requested "
                "spacing.\n\n"
                "```sh\n"
                "pip install 'polydisc[dashboard]'\n"
                "polydisc generate --bounds 500,500 --radii 2,4 --seed 0\n"
                "```"
            )
        return

    inferred = bool(st.session_state.get(_INFERRED_KEY, False))
    summary_metrics(pack)

    view, distribution, neighbours, full_report, download = st.tabs(
        ["View", "Size distribution", "Neighbours", "Report", "Download"]
    )
    with view:
        view_tab(pack)
    with distribution:
        distribution_tab(pack, inferred)
    with neighbours:
        neighbours_tab(pack)
    with full_report:
        report_tab(pack)
    with download:
        download_tab(pack)


if __name__ == "__main__":
    main()
