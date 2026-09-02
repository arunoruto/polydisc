"""Streamlit inspector for polydisc packings.

Run with ``streamlit run src/polydisc/app_streamlit.py``.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import streamlit as st
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from stpyvista import stpyvista

from polydisc.diagnostics import report
from polydisc.radii import RadiusSpec
from polydisc.regions import Ball, Box, Region
from polydisc.sampling import Packing, sample
from polydisc.spacing import spacing_scale


def _radius_spec_controls() -> RadiusSpec:
    """Render the distribution controls and return the chosen spec."""
    family = st.selectbox(
        "radius distribution",
        ["discrete", "lognormal", "normal", "uniform"],
        help="Discrete picks from a fixed list; the others are truncated "
        "continuous distributions sampled inside the kernel.",
    )

    if family == "lognormal":
        gmean = st.number_input("geometric mean radius", 0.01, value=1.0, step=0.1)
        gstd = st.number_input("geometric std (>=1)", 1.0, value=1.3, step=0.05)
        return RadiusSpec.lognormal(float(gmean), float(gstd))

    if family == "normal":
        mean = st.number_input("mean radius", 0.01, value=1.0, step=0.1)
        std = st.number_input("std of radius", 0.0, value=0.2, step=0.05)
        return RadiusSpec.normal(float(mean), float(std))

    rmin = st.number_input("min radius", 0.01, value=1.0, step=0.1)
    rmax = st.number_input("max radius", 0.01, value=2.0, step=0.1)

    if family == "uniform":
        return RadiusSpec.uniform(float(rmin), float(rmax))

    rcount = st.slider("# allowed radii", 1, 20, 3)
    values = np.linspace(float(rmin), float(rmax), int(rcount))
    weighting = st.selectbox("weights", ["uniform", "gaussian"])
    if weighting == "gaussian":
        centre = st.number_input(
            "weight centre", value=float(np.mean(values)), step=0.1
        )
        sigma = st.number_input("weight sigma", 0.001, value=0.5, step=0.1)
        z = (values - float(centre)) / float(sigma)
        return RadiusSpec.discrete(values, np.exp(-0.5 * z * z))
    return RadiusSpec.discrete(values)


def _download_buttons(pack: Packing) -> None:
    """Offer the packing as CSV and NPZ downloads."""
    col_csv, col_npz = st.columns(2)

    header = "x,y,r" if pack.dimensions == 2 else "x,y,z,r"
    csv_buf = io.StringIO()
    csv_buf.write(header + "\n")
    for point, radius in zip(pack.points, pack.radii):
        csv_buf.write(
            ",".join([*(f"{float(v):.10g}" for v in point), f"{float(radius):.10g}"])
        )
        csv_buf.write("\n")

    with col_csv:
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"polydisc_{pack.dimensions}d.csv",
            mime="text/csv",
        )

    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf, points=pack.points, radii=pack.radii, bounds=pack.bounds
    )
    npz_buf.seek(0)
    with col_npz:
        st.download_button(
            "Download NPZ",
            data=npz_buf,
            file_name=f"polydisc_{pack.dimensions}d.npz",
            mime="application/octet-stream",
        )


def _plot_2d(pack: Packing, show_discs: bool) -> None:
    """Draw a 2D packing with matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0.0, float(pack.bounds[0]))
    ax.set_ylim(0.0, float(pack.bounds[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelbottom=False, labelleft=False)

    if isinstance(pack.region, Ball):
        ax.add_patch(
            Circle(
                tuple(pack.region.centre),
                pack.region.radius,
                fill=False,
                edgecolor="0.4",
                linewidth=1.0,
            )
        )

    if show_discs:
        # One PatchCollection, not N patches: an artist per particle is
        # what makes a 100k-particle plot unusable.
        circles = [
            Circle((float(p[0]), float(p[1])), float(r))
            for p, r in zip(pack.points, pack.radii)
        ]
        collection = PatchCollection(circles, cmap="viridis", alpha=0.55)
        collection.set_array(pack.radii)
        ax.add_collection(collection)
        fig.colorbar(collection, ax=ax, label="radius")
    else:
        scatter = ax.scatter(
            pack.points[:, 0],
            pack.points[:, 1],
            c=pack.radii,
            s=8.0,
            cmap="viridis",
            linewidths=0.0,
        )
        fig.colorbar(scatter, ax=ax, label="radius")

    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _plot_3d(pack: Packing, resolution: int) -> None:
    """Draw a 3D packing with PyVista sphere glyphs."""
    plotter = pv.Plotter(window_size=[900, 700], off_screen=True)

    if isinstance(pack.region, Ball):
        outline = pv.Sphere(radius=pack.region.radius, center=tuple(pack.region.centre))
    else:
        outline = pv.Cube(
            center=tuple(float(b) / 2.0 for b in pack.bounds),
            x_length=float(pack.bounds[0]),
            y_length=float(pack.bounds[1]),
            z_length=float(pack.bounds[2]),
        )
    plotter.add_mesh(outline, style="wireframe", color="white", opacity=0.35)

    cloud = pv.PolyData(pack.points)
    cloud["radius"] = pack.radii
    # Glyphing builds one mesh for the whole packing; adding a sphere
    # per particle costs seconds once N passes a few thousand.
    geom = pv.Sphere(
        theta_resolution=int(resolution), phi_resolution=max(8, resolution // 2)
    )
    glyphed = cloud.glyph(scale="radius", geom=geom, orient=False, tolerance=0.0)
    plotter.add_mesh(glyphed, scalars="radius", cmap="viridis", opacity=0.9)

    plotter.set_background("#1f2430")
    plotter.view_isometric()
    stpyvista(plotter, use_container_width=True, key="pyvista_view")


def _plot_radius_histogram(pack: Packing) -> None:
    """Compare the realised radii against the requested distribution."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    unique = np.unique(pack.radii)
    if unique.size <= 40:
        ax.hist(pack.radii, bins=max(unique.size * 3, 5), color="#008cff", alpha=0.75)
        ax.set_xticks(unique)
    else:
        ax.hist(
            pack.radii,
            bins=max(20, int(np.sqrt(pack.n))),
            color="#008cff",
            alpha=0.75,
            density=True,
        )
        values, pmf = pack.spec.discretize(128)
        width = values[1] - values[0] if values.size > 1 else 1.0
        ax.plot(values, pmf / width, color="#ff6a00", lw=2, label="requested")
        ax.legend()

    ax.set_title("Realised radii")
    ax.set_xlabel("radius")
    ax.set_ylabel("count")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def main() -> None:
    """Run the Streamlit inspector."""
    st.set_page_config(page_title="polydisc inspector", layout="wide")

    # Helps VTK/PyVista render on headless systems; a no-op on desktops,
    # and not worth failing the app over when it is unavailable.
    with contextlib.suppress(Exception):
        pv.start_xvfb()

    st.title("polydisc")
    st.caption("Polydisperse Poisson-disc sampling for discs and spheres.")

    with st.sidebar:
        mode = st.radio(
            "container",
            ["2D box", "3D box", "ball", "carpet (thin box)"],
            index=0,
            help="A carpet is not a separate shape -- it is a box whose "
            "depth is about one particle diameter.",
        )

        if mode == "ball":
            ball_radius = st.number_input("ball radius", 1.0, value=60.0, step=5.0)
            ball_dims = st.radio("dimensions", [2, 3], index=1, horizontal=True)
        elif mode == "2D box":
            width = st.number_input("width", 1.0, value=200.0, step=10.0)
            height = st.number_input("height", 1.0, value=150.0, step=10.0)
        else:
            width = st.number_input("x extent", 1.0, value=100.0, step=10.0)
            height = st.number_input("y extent", 1.0, value=100.0, step=10.0)
            if mode == "3D box":
                depth = st.number_input("z extent", 1.0, value=100.0, step=10.0)

        spec = _radius_spec_controls()

        spacing = st.slider(
            "spacing (mean radii between neighbours)",
            0.0,
            6.0,
            0.0,
            0.25,
            help="2 leaves room for one average particle between "
            "neighbouring surfaces. Positions are scaled; radii are not.",
        )
        fit_bounds = st.checkbox(
            "keep requested size",
            value=True,
            help="Off: the container grows by the spacing factor instead "
            "(PyFracVAL behaviour).",
        )

        if mode == "carpet (thin box)":
            auto = st.checkbox("thinnest monolayer", value=True)
            if auto:
                # A monolayer needs its depth pre-inflated by the spacing
                # factor, or the pre-shrink leaves no room for a particle.
                inflate = spacing_scale(spacing) if fit_bounds else 1.0
                depth = 2.05 * spec.rmax * inflate
                st.caption(f"depth = {depth:.4g}")
            else:
                depth = st.number_input("depth", 0.01, value=10.0, step=1.0)

        repetitions = st.slider("repetitions (k)", 1, 60, 30)
        seed = int(st.number_input("seed", 0, 10_000_000, 0, 1))

        show_discs = st.checkbox("draw discs", value=True)
        resolution = 14
        if mode != "2D box":
            resolution = st.slider("sphere resolution", 8, 30, 14, 2)

        run = st.button("Generate", type="primary")

    if not run:
        st.info("Adjust the settings and press Generate.")
        return

    if mode == "ball":
        region: Region = Ball(ball_radius, ball_dims)
    elif mode == "2D box":
        region = Box([width, height])
    else:
        region = Box([width, height, depth])

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

    rep = report(pack)

    columns = st.columns(4)
    columns[0].metric("particles", pack.n)
    columns[1].metric("fill fraction", f"{pack.fill_fraction:.3f}")
    if pack.dimensions == 3:
        columns[2].metric("projected area", f"{rep.projected_area_fraction:.3f}")
    columns[3].metric("mean radius", f"{rep.radii['mean']:.4g}")

    if rep.gaps["overlaps"]:
        st.error(
            f"{rep.gaps['overlaps']} particles overlap a neighbour "
            f"(worst {rep.gaps['worst_overlap']:.3g})."
        )
    else:
        st.success("No overlaps.")

    if abs(rep.boundary_bias.get("bias") or 0.0) > 0.05:
        st.warning(
            "The container geometry is choosing the sizes: it shifts the "
            f"mean radius {rep.boundary_bias['bias'] * 100:+.1f}% before any "
            "crowding. Enlarge it or narrow the radius range."
        )

    if pack.n == 0:
        st.info("Nothing was placed.")
        return

    _download_buttons(pack)

    if pack.dimensions == 2:
        _plot_2d(pack, show_discs)
    else:
        _plot_3d(pack, resolution)

    _plot_radius_histogram(pack)

    with st.expander("Full report", expanded=False):
        st.code(str(rep))
        st.json(_jsonify(rep.to_dict()))


def _jsonify(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats for st.json."""
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


if __name__ == "__main__":
    main()
