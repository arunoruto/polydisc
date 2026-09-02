# polydisc dashboard

A Streamlit app for generating packings and inspecting ones you already
have. Streamlit is an **optional** dependency of `polydisc` — the
library installs without it.

## Run it locally

```sh
uv sync                                 # the dev group pulls the extra in
streamlit run dashboard/app.py
```

Or, from an installed copy:

```sh
pip install 'polydisc[dashboard]'
streamlit run dashboard/app.py
```

## What it does

**Generate** — pick a container (2D/3D box, 2D/3D ball, or a carpet
preset that sizes a thin box for you), a size distribution (lognormal,
discrete, normal, uniform), a spacing, and a seed.

**Upload a file** — drop in a `.npz` or `.csv` from `polydisc generate`
and get the same views. An `.npz` carries the full run settings, so the
requested-vs-realised comparison is meaningful; a `.csv` has only
coordinates and radii, so the container and size distribution are
inferred from the data and the comparison is suppressed rather than
faked.

Five tabs: the packing itself (true-scale 2D, interactive 3D point
cloud, or a true-scale cross-section of a 3D packing), the realised size
distribution against the requested one, nearest-neighbour clearances
against the requested spacing, the full report, and downloads.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app
   pointing at this repo with **main file path** `dashboard/app.py`.
3. That is all — `dashboard/requirements.txt` is picked up
   automatically.

Three details make this work, and are worth knowing before you change
anything:

- **The requirements file lives here, not in the repo root.** Community
  Cloud searches the entrypoint's directory first, and a dependency file
  there takes precedence over the root. This matters because Community
  Cloud does not read `uv.lock`, and it assumes any `pyproject.toml` it
  finds is Poetry format — which this project is not. Without
  `dashboard/requirements.txt` the build fails on the root
  `pyproject.toml`.
- **`polydisc` is not installed on Cloud.** `app.py` puts `../src` on
  `sys.path`, so the deployed app runs the repository's own source. That
  is why `requirements.txt` also lists the library's runtime
  dependencies; `tests/test_dashboard.py` fails if the two drift apart.
- **No `packages.txt` is needed.** The app renders with Matplotlib and
  Plotly only. An earlier version used PyVista/VTK, which needs system
  OpenGL libraries and a virtual framebuffer — avoidable complexity for
  a packing viewer.

`.streamlit/config.toml` in the repo root sets the accent colour and
opts out of usage stats. It deliberately does not pin light or dark:
the figures are drawn on transparent backgrounds so they follow whatever
the viewer has set.

## Layout

```
dashboard/
    app.py             entrypoint: sidebar, tabs, session state
    views.py           figures (Matplotlib for circles, Plotly for 3D)
    loaders.py         uploaded .npz / .csv -> Packing
    requirements.txt   Community Cloud dependency file
```
