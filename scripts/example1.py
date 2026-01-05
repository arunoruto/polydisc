import numpy as np
import plotly.colors as clrs
import plotly.graph_objects as go

from pysson_disc.poisson_disc import poisson_disc_samples

width = 1920
height = 1080

bounds = np.array([width, height])
radii = np.array([10, 20, 30]) * 5
points, radii = poisson_disc_samples(bounds, radii, 20)

fig = go.Figure()
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
)
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers"))
for p, r in zip(points, radii):
    # print(p[0], p[1], r)
    x0 = p[0] - r
    y0 = p[1] - r
    x1 = p[0] + r
    y1 = p[1] + r
    color = clrs.sample_colorscale(
        "viridis", np.interp(r, [np.min(radii), np.max(radii)], [0, 1]), colortype="rgb"
    )[0]
    # print(x0, y0, x1, y1)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=p[0] - r,
        y0=p[1] - r,
        x1=p[0] + r,
        y1=p[1] + r,
        line_color=color,
        fillcolor=color,
        opacity=0.5,
    )

fig.add_trace(
    go.Scatter(
        x=[0, width, width, 0, 0],
        y=[0, 0, height, height, 0],
        mode="lines",
        line=dict(color="black", width=1),
    )
)

cell_size = 2 * np.min(radii) / np.sqrt(len(bounds))
fig.update_layout(
    xaxis=dict(dtick=cell_size, showticklabels=False),
    yaxis=dict(scaleanchor="x", scaleratio=1, dtick=cell_size, showticklabels=False),
    showlegend=False,
)
# fig.show()
fig.write_image("out/example1.png", width=width, height=height, scale=2)
fig.write_html("out/example1.html")
