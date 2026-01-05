from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from poisson_disc import poisson_disc_samples


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
            "marker size", min_value=1, max_value=12, value=4 if dims == 2 else 3
        )
        scale_marker_by_radius = st.checkbox(
            "scale marker size by radius", value=(dims == 3)
        )
        max_spheres = st.slider(
            "max 3D spheres", min_value=0, max_value=300, value=100, step=10
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
        fig = go.Figure()

        if show_discs:
            for p, r in zip(points, pradii):
                fig.add_shape(
                    type="circle",
                    x0=float(p[0] - r),
                    y0=float(p[1] - r),
                    x1=float(p[0] + r),
                    y1=float(p[1] + r),
                    line=dict(width=1, color="rgba(0,0,0,0.35)"),
                    fillcolor="rgba(0,140,255,0.15)",
                )

        marker_sizes = (
            pradii / float(np.max(pradii)) * (marker_size * 2.5)
            if (scale_marker_by_radius and len(points) > 0)
            else np.full(len(points), float(marker_size))
        )

        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_sizes,
                    color=pradii,
                    colorscale="Viridis",
                    showscale=True,
                    line=dict(width=0),
                ),
                name="samples",
            )
        )

        if show_cells:
            cell_size = 2.0 * float(np.min(radii)) / np.sqrt(dims)
            fig.update_xaxes(dtick=cell_size)
            fig.update_yaxes(dtick=cell_size)
        else:
            # Let Plotly choose sensible ticks
            fig.update_xaxes(dtick=None)
            fig.update_yaxes(dtick=None)

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            height=700,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, float(bounds[0])]),
            yaxis=dict(range=[0, float(bounds[1])]),
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch")

        if len(points) > 0:
            hist = go.Figure(
                data=[
                    go.Histogram(
                        x=pradii,
                        nbinsx=max(1, int(len(np.unique(radii)))),
                        marker_color="rgba(0,140,255,0.65)",
                    )
                ]
            )
            hist.update_layout(
                title="Picked radii distribution",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="radius",
                yaxis_title="count",
            )
            st.plotly_chart(hist, width="stretch")

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=(
                        (pradii / float(np.max(pradii)) * (marker_size * 2.0) + 1.0)
                        if (scale_marker_by_radius and len(points) > 0)
                        else float(marker_size)
                    ),
                    color=pradii,
                    colorscale="Viridis",
                    opacity=0.9,
                ),
                name="samples",
            )
        )

        if show_discs:
            for p, r in list(zip(points, pradii))[: int(max_spheres)]:
                xs, ys, zs = _sphere_points(
                    p, float(r), resolution=int(sphere_resolution)
                )
                fig.add_trace(
                    go.Surface(
                        x=xs,
                        y=ys,
                        z=zs,
                        opacity=0.15,
                        showscale=False,
                        surfacecolor=np.zeros_like(xs),
                        colorscale=[
                            [0.0, "rgba(0,140,255,0.15)"],
                            [1.0, "rgba(0,140,255,0.15)"],
                        ],
                        name="sphere",
                    )
                )

        fig.update_layout(
            height=750,
            margin=dict(l=20, r=20, t=40, b=20),
            scene=dict(
                xaxis=dict(range=[0, float(bounds[0])]),
                yaxis=dict(range=[0, float(bounds[1])]),
                zaxis=dict(range=[0, float(bounds[2])]),
                aspectmode="cube",
            ),
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch")

        if len(points) > 0:
            hist = go.Figure(
                data=[
                    go.Histogram(
                        x=pradii,
                        nbinsx=max(1, int(len(np.unique(radii)))),
                        marker_color="rgba(0,140,255,0.65)",
                    )
                ]
            )
            hist.update_layout(
                title="Picked radii distribution",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="radius",
                yaxis_title="count",
            )
            st.plotly_chart(hist, width="stretch")

        st.caption("Note: Rendering spheres is expensive; shows first N from sidebar.")


if __name__ == "__main__":
    main()
