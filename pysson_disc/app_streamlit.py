from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import streamlit as st
from matplotlib.patches import Circle
from stpyvista import stpyvista

from pysson_disc.poisson_disc import poisson_disc_samples


def _find_violations(points: np.ndarray, radii: np.ndarray) -> tuple[int, float]:
    if len(points) <= 1:
        return 0, 0.0

    diffs = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=-1)
    min_dist = radii[:, None] + radii[None, :]
    min_dist2 = min_dist * min_dist

    mask = ~np.eye(len(points), dtype=bool)
    violations = dist2[mask] < min_dist2[mask]
    count = int(np.count_nonzero(violations))

    if count == 0:
        return 0, 0.0

    overlap_amount = np.sqrt(min_dist2[mask][violations]) - np.sqrt(
        dist2[mask][violations]
    )
    return count, float(np.max(overlap_amount))


def _sphere_points(
    center: np.ndarray, radius: float, resolution: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, max(8, resolution // 2))
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main() -> None:
    st.set_page_config(page_title="Poisson-disc inspector", layout="wide")

    # Helps VTK/PyVista render inside Streamlit on headless systems.
    # On desktop setups this is typically a no-op.
    try:
        pv.start_xvfb()
    except Exception:
        pass

    st.title("Poisson-disc sampling inspector")
    st.caption("Interactive view for 2D/3D box sampling (Numba accelerated).")

    with st.sidebar:
        dims = st.radio("Dimensions", options=[2, 3], index=0)
        repetitions = st.slider(
            "repetitions (k)", min_value=1, max_value=60, value=20, step=1
        )
        seed = st.number_input(
            "random seed", min_value=0, max_value=10_000_000, value=0, step=1
        )

        if dims == 2:
            width = st.number_input("width", min_value=10.0, value=800.0, step=10.0)
            height = st.number_input("height", min_value=10.0, value=500.0, step=10.0)
            bounds = np.array([width, height], dtype=np.float64)
        else:
            x = st.number_input("x", min_value=10.0, value=200.0, step=10.0)
            y = st.number_input("y", min_value=10.0, value=200.0, step=10.0)
            z = st.number_input("z", min_value=10.0, value=200.0, step=10.0)
            bounds = np.array([x, y, z], dtype=np.float64)

        rmin = st.number_input("min radius", min_value=0.1, value=8.0, step=0.5)
        rmax = st.number_input("max radius", min_value=0.1, value=16.0, step=0.5)
        rcount = st.slider(
            "# allowed radii", min_value=1, max_value=20, value=5, step=1
        )

        radii = np.linspace(float(rmin), float(rmax), int(rcount), dtype=np.float64)
        show_cells = st.checkbox("show grid", value=(dims == 2))
        show_discs = st.checkbox("show discs/spheres", value=True)

        marker_size = st.slider(
            "marker size",
            min_value=1,
            max_value=12,
            value=4 if dims == 2 else 3,
            help=("Point marker size (2D), or 3D point markers when spheres are off."),
        )
        sphere_resolution = st.slider(
            "3D sphere resolution", min_value=8, max_value=30, value=14, step=2
        )

        run = st.button("Generate", type="primary")

    if not run:
        st.info("Adjust settings and click Generate.")
        return

    np.random.seed(int(seed))

    with st.spinner("Generating samples..."):
        points, pradii = poisson_disc_samples(
            bounds, radii, repetitions=int(repetitions)
        )

    st.success(f"Generated {len(points)} samples")

    if len(points) > 0:
        col_csv, col_npz = st.columns(2)

        csv_buf = io.StringIO()
        if dims == 2:
            csv_buf.write("x,y,r\n")
        else:
            csv_buf.write("x,y,z,r\n")

        for p, r in zip(points, pradii):
            row = ",".join([*(f"{float(v):.10g}" for v in p), f"{float(r):.10g}"])
            csv_buf.write(row + "\n")

        with col_csv:
            st.download_button(
                "Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"poisson_disc_{dims}d.csv",
                mime="text/csv",
            )

        npz_buf = io.BytesIO()
        np.savez_compressed(
            npz_buf, points=points, radii=pradii, bounds=bounds, allowed_radii=radii
        )
        npz_buf.seek(0)

        with col_npz:
            st.download_button(
                "Download NPZ",
                data=npz_buf,
                file_name=f"poisson_disc_{dims}d.npz",
                mime="application/octet-stream",
            )

    violation_count, worst_overlap = _find_violations(points, pradii)
    if violation_count > 0:
        st.error(
            f"Found {violation_count} pairwise overlap checks failing. "
            f"Worst overlap: {worst_overlap:.6g}"
        )
    else:
        st.success("No overlaps detected (pairwise check).")

    with st.expander("Stats", expanded=False):
        st.write(
            {
                "dims": int(dims),
                "bounds": bounds.tolist(),
                "repetitions": int(repetitions),
                "seed": int(seed),
                "allowed_radii_min": float(np.min(radii)),
                "allowed_radii_max": float(np.max(radii)),
                "samples": int(len(points)),
            }
        )
        if len(points) > 0:
            st.write(
                {
                    "picked_radii_min": float(np.min(pradii)),
                    "picked_radii_max": float(np.max(pradii)),
                    "picked_radii_mean": float(np.mean(pradii)),
                }
            )

    if dims == 2:
        if len(points) == 0:
            st.info("No samples to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlim(0.0, float(bounds[0]))
        ax.set_ylim(0.0, float(bounds[1]))
        ax.set_aspect("equal", adjustable="box")

        if show_cells:
            cell_size = 2.0 * float(np.min(radii)) / np.sqrt(dims)
            ax.set_xticks(np.arange(0.0, float(bounds[0]) + cell_size, cell_size))
            ax.set_yticks(np.arange(0.0, float(bounds[1]) + cell_size, cell_size))
            ax.grid(True, linewidth=0.5, alpha=0.25)
            ax.tick_params(
                axis="both", which="both", labelbottom=False, labelleft=False
            )
        else:
            ax.tick_params(
                axis="both", which="both", labelbottom=False, labelleft=False
            )

        allowed = np.unique(pradii)
        cmap = plt.get_cmap("viridis", allowed.shape[0])

        if show_discs:
            for p, r in zip(points, pradii):
                idx = int(np.searchsorted(allowed, r))
                cr, cg, cb, _ = cmap(idx)
                ax.add_patch(
                    Circle(
                        (float(p[0]), float(p[1])),
                        float(r),
                        facecolor=(float(cr), float(cg), float(cb), 0.22),
                        edgecolor="none",
                        linewidth=0.0,
                    )
                )
        else:
            sc = ax.scatter(
                points[:, 0],
                points[:, 1],
                c=pradii,
                s=float(marker_size) ** 2 * 8.0,
                cmap="viridis",
                linewidths=0.0,
            )
            fig.colorbar(sc, ax=ax, label="radius")

        st.pyplot(fig, use_container_width=True)

        # Histogram (discrete radii friendly)
        fig_h, ax_h = plt.subplots(figsize=(10, 2.5))
        unique_r = np.unique(pradii)
        if unique_r.shape[0] <= 100:
            # Bin edges halfway between consecutive discrete radii.
            half_steps = (unique_r[:-1] + unique_r[1:]) * 0.5
            left = (
                unique_r[0] - (half_steps[0] - unique_r[0])
                if unique_r.shape[0] > 1
                else unique_r[0] - 0.5
            )
            right = (
                unique_r[-1] + (unique_r[-1] - half_steps[-1])
                if unique_r.shape[0] > 1
                else unique_r[-1] + 0.5
            )
            bins = np.concatenate([[left], half_steps, [right]]).tolist()
            ax_h.hist(pradii, bins=bins, color="#008cff", alpha=0.7, rwidth=0.9)
            ax_h.set_xticks(unique_r)
        else:
            ax_h.hist(
                pradii,
                bins=max(20, int(np.sqrt(len(pradii)))),
                color="#008cff",
                alpha=0.7,
            )
        ax_h.set_title("Picked radii distribution")
        ax_h.set_xlabel("radius")
        ax_h.set_ylabel("count")
        st.pyplot(fig_h, use_container_width=True)

    else:
        if len(points) == 0:
            st.info("No samples to plot.")
            return

        plotter = pv.Plotter(window_size=[900, 700], off_screen=True)

        # Bounds box
        cube = pv.Cube(
            center=(bounds[0] / 2.0, bounds[1] / 2.0, bounds[2] / 2.0),
            x_length=bounds[0],
            y_length=bounds[1],
            z_length=bounds[2],
        )
        plotter.add_mesh(cube, style="wireframe", color="white", opacity=0.35)

        if show_discs:
            allowed = np.unique(pradii)
            cmap = plt.get_cmap("viridis", allowed.shape[0])

            for p, r in zip(points, pradii):
                idx = int(np.searchsorted(allowed, r))
                cr, cg, cb, _ = cmap(idx)

                sphere = pv.Sphere(
                    radius=float(r),
                    center=(float(p[0]), float(p[1]), float(p[2])),
                    theta_resolution=int(sphere_resolution),
                    phi_resolution=max(8, int(sphere_resolution // 2)),
                )
                plotter.add_mesh(
                    sphere,
                    color=(float(cr), float(cg), float(cb)),
                    opacity=0.15,
                    style="surface",
                )
        else:
            cloud = pv.PolyData(points)
            cloud["r"] = pradii
            plotter.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=float(marker_size) * 2.0,
                scalars="r",
                cmap="viridis",
                opacity=1.0,
            )

        plotter.set_background("#1f2430")
        plotter.view_isometric()

        stpyvista(plotter, use_container_width=True, key="pyvista_view")

        fig_h, ax_h = plt.subplots(figsize=(10, 2.5))
        ax_h.hist(
            pradii,
            bins=max(5, int(len(np.unique(radii)) * 2)),
            color="#008cff",
            alpha=0.7,
        )
        ax_h.set_title("Picked radii distribution")
        ax_h.set_xlabel("radius")
        ax_h.set_ylabel("count")
        st.pyplot(fig_h, use_container_width=True)

        st.caption("Note: Rendering spheres can still be expensive.")


if __name__ == "__main__":
    main()
