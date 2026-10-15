<a id="readme-top"></a>

[![PyPI][pypi-shield]][pypi-url]
[![Tests][tests-shield]][tests-url]
[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <h3 align="center">polydisc</h3>

  <p align="center">
    Polydisperse Poisson-disc sampling for discs and spheres
    <br />
    <a href="https://github.com/arunoruto/polydisc"><strong>Explore the repository »</strong></a>
    <br />
    <br />
    <a href="https://pypi.org/project/polydisc/">PyPI</a>
    &middot;
    <a href="https://github.com/arunoruto/polydisc/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/arunoruto/polydisc/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#quickstart">Quickstart</a></li>
        <li><a href="#regions">Regions</a></li>
        <li><a href="#periodic-boxes-and-a-minimum-gap">Periodic boxes and a minimum gap</a></li>
        <li><a href="#flat-carpets">Flat carpets</a></li>
        <li><a href="#spacing">Spacing</a></li>
        <li><a href="#radius-distributions">Radius distributions</a></li>
        <li><a href="#diagnostics-what-you-actually-got">Diagnostics</a></li>
        <li><a href="#command-line">Command line</a></li>
        <li><a href="#dashboard">Dashboard</a></li>
        <li><a href="#performance">Performance</a></li>
        <li><a href="#implementation-notes">Implementation notes</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## About The Project

A Numba-accelerated, variable-radius variant of Bridson's algorithm
(*Fast Poisson disk sampling in arbitrary dimensions*, SIGGRAPH 2007,
algorithm 4.3) in 2D and 3D: every particle gets its own radius drawn
from a distribution you choose, no two particles overlap, and none pokes
outside the container -- a box or a ball. Boxes can be periodic along
any axis, so a packing can be one cell of an infinite (or laterally
infinite) medium, and a minimum surface gap between particles can be
enforced. Built for generating packings for light-scattering work, so it
also tells you what it *actually* produced, not just what you asked for.

Formerly `pysson-disc`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][python-shield]][python-url]
* [![NumPy][numpy-shield]][numpy-url]
* [![Numba][numba-shield]][numba-url]
* [![Streamlit][streamlit-shield]][streamlit-url] (optional, for the dashboard)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Getting Started

### Prerequisites

* Python 3.13 or newer
* For working on the code: [uv](https://docs.astral.sh/uv/) (optionally
  [devenv](https://devenv.sh); `direnv allow` picks it up automatically)

### Installation

From PyPI:

```sh
pip install polydisc                  # the sampler
pip install "polydisc[dashboard]"     # ... plus the Streamlit dashboard
```

Streamlit is an optional extra: the library itself pulls in nothing but
NumPy, Numba, Click and PyYAML.

From source, with uv:

```sh
git clone https://github.com/arunoruto/polydisc.git
cd polydisc
uv sync                # the dev group includes the dashboard extra
uv run pytest          # tests, including the examples in this README
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Usage

### Quickstart

```python
>>> import polydisc
>>> pack = polydisc.sample([200.0, 150.0], radii=[2.0, 4.0], seed=0)
>>> pack.points.shape[1], pack.n == len(pack.radii)
(2, True)
>>> 0.4 < pack.fill_fraction < 0.7
True

```

The same run from the shell:

```sh
polydisc generate --bounds 200,150 --radii 2,4 --seed 0 --out pack.npz
```

A `Packing` carries the geometry *and* the settings that produced it, so
a result is always replayable:

```python
>>> replay = polydisc.sample([200.0, 150.0], radii=[2.0, 4.0], seed=pack.seed)
>>> bool((replay.points == pack.points).all())
True

```

### Regions

Particles fill a **region**. Pass a sequence of edge lengths and you get
a `Box`; pass a `Ball` to fill a sphere instead.

```python
>>> box = polydisc.sample(polydisc.Box([300.0, 200.0]), 4.0, seed=0)
>>> ball = polydisc.sample(polydisc.Ball(150.0, dimensions=2), 4.0, seed=0)
>>> box.region.label, ball.region.label
('box, 300 x 200', 'ball, radius 150')
>>> bool(ball.region.contains(ball.points, ball.radii).all())
True

```

| Region                                 | Container                                   |
| -------------------------------------- | ------------------------------------------- |
| `Box([lx, ly])`                        | 2D rectangle                                |
| `Box([lx, ly, lz])`                    | 3D box — a **carpet** is one of these, thin |
| `Box([lx, ly, lz], periodic=(True, True, False))` | slab, laterally infinite         |
| `Ball(r, dimensions=2)`                | filled disc                                 |
| `Ball(r)`                              | filled sphere                               |

It is called `Ball` rather than `Sphere` because the particles inside it
are spheres too; this is the container they go in.

```sh
polydisc generate --bounds 500,500 --radii 2,4 --seed 0
polydisc generate --shape ball --ball-radius 200 --radii 2,4 --seed 0
```

Every region reports its `volume`, `footprint`, `bounding_box` and
`max_particle_radius`, and can `scaled()` itself — that is all the
sampler needs, so a new shape is a small addition.

### Periodic boxes and a minimum gap

`Box(..., periodic=...)` makes chosen axes wrap around, one flag per
axis (or a single bool for all). Along a periodic axis a candidate that
leaves one face re-enters through the opposite one, like the snake in
the game, and every distance -- the neighbour grid, the overlap test,
the gap -- uses the minimum-image convention. Copies of the box then tile
space without overlaps. The other axes keep hard walls that particles
stay inside. Balls cannot be periodic.

`gap` sets a minimum surface-to-surface distance, in the units of the
radii: every pair ends up with `|c_i - c_j| >= r_i + r_j + gap`,
across periodic faces too. Unlike [spacing](#spacing), which dilates a
finished packing, the gap is a hard constraint while sampling, so the
packing still saturates; walls are not affected by it.

A slab that is periodic in x and y with hard top and bottom faces,
lognormal radii truncated to `[1/3, 5/3]` and a gap of a quarter of the
median radius:

```python
>>> import math
>>> slab = polydisc.Box([32.0, 32.0, 44.0], periodic=(True, True, False))
>>> spec = polydisc.RadiusSpec.lognormal(
...     gmean=1.0, gstd=math.exp(0.3), rmin=1 / 3, rmax=5 / 3
... )
>>> pack = polydisc.sample(slab, spec, gap=0.25, seed=0)
>>> pack.region.label
'box, 32 x 32 x 44, periodic x, y'
>>> gaps = polydisc.nearest_neighbour_gaps(pack.points, pack.radii, pack.region)
>>> bool(gaps.min() >= 0.25 - 1e-9)
True
>>> 0.2 < pack.fill_fraction < 0.25
True

```

That takes well under a second (about 3900 spheres, 23 % filled; with
`gap=1.0`, about 1300 spheres and 9 %). Passing the region to
`nearest_neighbour_gaps`, `gap_report` or `report` makes them measure
across the periodic faces. A periodic edge must be at least
`2 * (2 * rmax + gap)` long, so that the nearest image of a neighbour is
unambiguous.

```sh
polydisc generate --bounds 32,32,44 --periodic xy --gap 0.25 \
    --dist lognormal --gmean 1 --gstd 1.35 --rmin 0.33 --rmax 1.67 --out slab.npz
```

### Flat carpets

A carpet is not a separate shape: it is a `Box` whose depth is about one
particle diameter. `2.05 * rmax` is the thinnest slab that still admits
the largest particle.

```python
>>> spec = polydisc.RadiusSpec.lognormal(gmean=1.0, gstd=1.3)
>>> carpet = polydisc.sample(
...     [300.0, 300.0, 2.05 * spec.rmax], spec, seed=0
... )
>>> carpet.dimensions
3
>>> bool(carpet.points[:, 2].max() <= carpet.bounds[2])
True

```

Spheres pack in genuine 3D, so they may overlap seen from above;
`projected_area_fraction` is the summed disc area over the container's
footprint, which is the quantity single-scattering estimates want (and
which can exceed 1 for a deeper deposit).

One wrinkle: with `fit_bounds` on, spacing pre-shrinks *every* axis, so
the depth has to be inflated by the same factor or the slab stops being
a monolayer.

```python
>>> spacing = 2.0
>>> depth = 2.05 * spec.rmax * polydisc.spacing_scale(spacing)
>>> spaced = polydisc.sample(
...     [300.0, 300.0, depth], spec, spacing=spacing, seed=0
... )
>>> spaced.n < carpet.n
True

```

### Spacing

Spacing is expressed as **how many mean radii of clearance sit between
neighbouring surfaces** — the same convention as
[PyFracVAL][pyfracval-url] and [YASF][yasf-url]. It is applied as a
position-only dilation; radii are never touched.

```
T = 1 + spacing / 2
positions *= T
```

| `spacing` | `T` | gap between two touching average particles |
| --------- | --- | ------------------------------------------ |
| 0         | 1.0 | 0 (as generated)                           |
| 1         | 1.5 | half an average radius                     |
| 2         | 2.0 | one average **particle**                   |
| 4         | 3.0 | two average particles                      |

```python
>>> import numpy as np
>>> from polydisc import apply_spacing
>>> touching = np.array([[0.0, 0.0], [6.0, 0.0]])   # two r=3 particles
>>> moved = apply_spacing(touching, spacing=2.0)
>>> float(np.linalg.norm(moved[1] - moved[0]) - 6.0)   # gap == 2 * r_mean
6.0

```

Because it is a single global dilation, every pairwise distance scales
by the same factor: the structure is stretched, not reshaped. For a hard
lower bound on every gap, use [`gap`](#periodic-boxes-and-a-minimum-gap)
instead (or as well: a `gap` requested together with `spacing` still
holds in the finished, dilated packing).

#### Where does the extra room come from?

`fit_bounds` decides whether the container or the particle count
absorbs the spacing.

```python
>>> fitted = polydisc.sample([200.0, 200.0], 4.0, spacing=2.0, seed=1)
>>> grown = polydisc.sample(
...     [200.0, 200.0], 4.0, spacing=2.0, fit_bounds=False, seed=1
... )
>>> fitted.bounds.tolist(), grown.bounds.tolist()
([200.0, 200.0], [400.0, 400.0])
>>> fitted.n < grown.n
True

```

* `fit_bounds=True` (default) samples a pre-shrunk container, then
  dilates, so the finished packing occupies exactly the container you
  asked for and holds fewer particles.
* `fit_bounds=False` samples the full box and lets the domain grow —
  PyFracVAL's behaviour, and the one that keeps the particle count.

### Radius distributions

| Constructor                                                 | Parameters                 |
| ----------------------------------------------------------- | -------------------------- |
| `RadiusSpec.discrete(values, weights=None)`                 | a fixed set of radii       |
| `RadiusSpec.lognormal(gmean, gstd, truncate=2, rmin, rmax)` | geometric mean and spread  |
| `RadiusSpec.normal(mean, std, truncate=2, rmin, rmax)`      | arithmetic mean and spread |
| `RadiusSpec.uniform(rmin, rmax)`                            | flat over a range          |

Continuous families are truncated and sampled inside the kernel; `rmin`
and `rmax` set the window explicitly. The lognormal defaults match
FracVAL's `±2 GSD` window, `[gmean / gstd², gmean · gstd²]`:

```python
>>> spec = polydisc.RadiusSpec.lognormal(gmean=10.0, gstd=1.5)
>>> round(spec.rmin, 3), round(spec.rmax, 3)
(4.444, 22.5)

```

Truncation preserves the geometric mean but lowers the realised spread
to about `gstd ** 0.88` — the same relation PyFracVAL documents. Draw
from a spec directly to see what the sampler sees:

```python
>>> draws = spec.sample(50_000, seed=0)
>>> bool(draws.min() >= spec.rmin and draws.max() <= spec.rmax)
True

```

A bare float or list is coerced for you: `sample(bounds, 3.0)` is
monodisperse, `sample(bounds, [1.0, 2.0])` is a discrete pair.

### Diagnostics: what you actually got

Two effects push the realised size distribution away from the request,
and both are reported rather than hidden.

**Boundary exclusion.** A particle of radius `r` can only be centred in
the shrunken box `b_d - 2r`, so big particles have fewer places to go.
Negligible in a roomy box, dominant in a thin slab — it is why a
"monolayer" of a wide distribution comes out almost entirely fine.
Periodic axes have no walls and exclude nothing. `geometric_bias`
predicts it before you sample:

```python
>>> two_sizes = polydisc.RadiusSpec.discrete([1.0, 3.0])
>>> round(polydisc.geometric_bias(two_sizes, [200.0, 200.0])["bias"], 3)
-0.01
>>> round(polydisc.geometric_bias(two_sizes, [200.0, 200.0, 6.2])["bias"], 3)
-0.456

```

`sample` logs a warning when this exceeds 5%. A ball and its bounding
cube score identically here: both admit a fraction `(1 - r/R)**d` of
their own measure.

**Crowding.** Candidate radii are drawn once per placement *attempt*, so
a small candidate survives a tight neighbourhood more often than a large
one. This is inherent to the dart-throwing formulation; no closed form
predicts it, so measure it:

```python
>>> pack = polydisc.sample([200.0, 200.0], two_sizes, seed=0)
>>> stats = polydisc.radius_report(pack.radii, pack.spec)
>>> stats["requested_mean"]
2.0
>>> bool(stats["mean_bias"] < 0)     # realised mean is smaller
True

```

`report` bundles everything — realised moments, both biases,
nearest-neighbour gaps, fill fractions:

```python
>>> print(polydisc.report(pack))         # doctest: +ELLIPSIS
particles         ...
dimensions        2D
region            box, 200 x 200
fill fraction     ...
radius            mean ...
radius (geom)     ...
vs requested      mean 2  bias ...
  of which walls  ...
neighbour gap     min ...
gap / mean radius min ...

```

`nearest_neighbour_gaps` gives the per-particle surface clearance and is
exact, not approximate — the shell search only stops once no unvisited
cell could hold anything closer.

### Command line

```
polydisc generate   fill a box or a ball
polydisc report     print diagnostics for a saved .npz
polydisc radii      print a discrete radius set for pasting into a config
```

```sh
polydisc generate --bounds 500,500,50 --dist lognormal --gmean 2 --gstd 1.3 \
    --spacing 2 --repetitions 30 --seed 0 --out pack.npz
polydisc generate --shape ball --ball-radius 200 --radii 1,2 --seed 0
polydisc generate --bounds 100,100,20 --periodic xy --gap 0.5 --radii 1,2
```

CSV goes to stdout when `--out` is omitted, with diagnostics on stderr
so a pipe stays clean:

```sh
polydisc generate --bounds 500,500,4.1 --radii 1,2 --seed 0 | head -5
```

Config files may be TOML, YAML or JSON, with optional named profiles;
command-line flags override them.

```toml
# packing.toml
seed = 0
repetitions = 30
dist = "lognormal"
gmean = 1.0
gstd = 1.3

[profiles.dense]
bounds = [500.0, 500.0, 4.1]
spacing = 0.0

[profiles.sparse]
bounds = [500.0, 500.0, 8.2]
spacing = 3.0

[profiles.slab]
bounds = [64.0, 64.0, 20.0]
periodic = "xy"
gap = 0.5

[profiles.ball]
shape = "ball"
ball_radius = 200.0
```

```sh
polydisc generate --config packing.toml --profile sparse --out sparse.npz
```

### Dashboard

```sh
pip install "polydisc[dashboard]"
streamlit run dashboard/app.py      # or: `dashboard` inside devenv shell
```

Generate packings interactively, or **upload a `.npz` or `.csv`** from
`polydisc generate` and inspect it: true-scale 2D views, an interactive
3D point cloud, true-scale cross-sections of a 3D packing, the realised
size distribution against the requested one, nearest-neighbour
clearances against the requested spacing, and downloads.

The dashboard lives in [`dashboard/`](dashboard/) as plain scripts, not
inside the package, so it can be deployed to
[Streamlit Community Cloud](https://share.streamlit.io) by pointing an
app at `dashboard/app.py`. See [`dashboard/README.md`](dashboard/README.md)
for the deployment details — in particular why the requirements file
has to live in that folder rather than the repo root.

### Performance

Against the previous implementation, same seeds (`scripts/benchmark.py`):

| case                          |     N |    old |    new | speedup |
| ----------------------------- | ----: | -----: | -----: | ------: |
| 2D 1000², r=5                 |  6.1k | 0.046 s | 0.017 s |    2.7× |
| 2D 4000², r=1..5              |  729k |  8.60 s |  3.59 s |    2.4× |
| 2D 500², r=0.25..5 (20×)      |   58k |  2.28 s |  0.55 s |    4.1× |
| 3D 150³, r=1..5               |   61k |  6.90 s |  1.60 s |    4.3× |
| 3D 60³, r=0.5..5 (10×)        |   24k |  9.89 s |  2.88 s |    3.4× |
| 3D 300³, r=1..5               |  483k | 66.0 s | 17.4 s |    3.8× |

Peak memory on the last row fell from 699 MB to 176 MB. Every row
also packs *more* particles than before, so the speedup is measured
against a denser result, not a cheaper one.

Two changes account for most of it:

* **Cell size.** The acceleration grid uses `cell = 2 · rmax + gap`,
  exactly the largest possible query radius, so the neighbour scan is
  always a 3×3 (or 3×3×3) block *whatever* the `rmax / rmin` ratio.
  Sizing cells by `rmin` instead — the textbook choice, which guarantees
  at most one point per cell — makes the scan grow as `(rmax/rmin)^d`
  and the grid memory as `(1/rmin)^d`. Cells now hold several points in
  an intrusive linked list, so memory scales with the number of
  *particles*, not with the number of grid cells. A ball is sampled on
  the grid of its bounding box, which just leaves the corner cells
  empty. Along a periodic axis the cells are widened so a whole number
  of them tiles the edge, and the scan wraps around the grid.
* **No allocation in the hot loop.** Candidate positions, directions and
  cell indices are all scalar arithmetic. A direct NumPy-style
  translation allocates several small arrays per candidate, which is
  where Numba spends its time.

### Implementation notes

* Candidates are placed in the annulus `[d, 2d]` around the active
  point, `d = r_i + r_j + gap` — Bridson's `[r, 2r]` generalised to two
  radii, with a uniform (not area-weighted) radial draw, which packs
  slightly denser.
* A candidate that lands outside the container is a wasted draw, not a
  failed placement, so it is redrawn rather than charged against
  Bridson's `k`. In a 3D box the depth is handled exactly instead: by
  Archimedes' hat-box theorem a uniform direction's `z` is uniform along
  the axis, so drawing `z` directly inside the slab samples precisely
  the in-slab directions with no rejection. Without this a monolayer
  stops at a single particle — in a slab `2.05 · r` deep a blind
  direction lands inside about one time in eighty.
* Seeding is done *inside* the kernel. Numba's generator is separate
  from NumPy's, so calling `np.random.seed` from Python does not reach
  it; pass `seed=` instead. Omitting it draws one from NumPy's global
  generator and records it on the result.
* With `gap=0` and no periodic axis the kernels compute exactly what
  they did before either option existed: seeded results are
  bit-identical to 0.3.0.
* The default `max_points` is the hard geometric ceiling
  `container / smallest particle`, which cannot truncate a real run.

Layout of the repository:

```
src/polydisc/
    _kernels.py       Numba kernels: samplers, radius draws, gap search
    regions.py        Box and Ball: the containers to fill
    radii.py          RadiusSpec: discrete and continuous distributions
    sampling.py       sample(), Packing, geometric_bias()
    spacing.py        spacing -> scale factor
    diagnostics.py    realised radii, gaps, report
    cli.py            polydisc command
dashboard/            Streamlit app (optional extra, deployable)
reference/            original MATLAB implementation
scripts/              example plots and the benchmark
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Roadmap

- [x] Box and ball containers in 2D and 3D
- [x] Discrete, lognormal, normal and uniform radius distributions
- [x] Realised-distribution, boundary-bias and gap diagnostics
- [x] Command line with TOML/YAML/JSON configs, and a Streamlit dashboard
- [x] Periodic box axes (minimum image)
- [x] Minimum surface gap between particles
- [ ] Position-dependent radius distributions, e.g. a mean radius that
      varies with depth `z`, to model size gradients such as those in
      space-weathered rims
- [ ] Documentation site

See the [open issues][issues-url] for a full list of proposed features
(and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contributing

Contributions are welcome. If you have a suggestion, open an issue with
the label "enhancement", or fork the repository and open a pull request:

1. Fork the project
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes with a [conventional commit](https://www.conventionalcommits.org)
   message (`git commit -m "feat: add an amazing feature"`) — releases
   and the changelog are generated from them
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a pull request

`uv run pytest` runs the tests and the examples in this README;
`uv run ruff check` and `uv run ruff format` keep the style.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more
information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contact

Mirza Arnaut - mirza.arnaut@tu-dortmund.de

Project link: [https://github.com/arunoruto/polydisc](https://github.com/arunoruto/polydisc)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Acknowledgments

* R. Bridson, [*Fast Poisson disk sampling in arbitrary dimensions*][bridson-url],
  ACM SIGGRAPH 2007 sketches — the algorithm this package generalises.
* G. Marsaglia, *Choosing a point from the surface of a sphere*,
  Ann. Math. Statist. 43 (1972) 645–646 — the direction draw in the
  ball sampler.
* [PyFracVAL][pyfracval-url] and FracVAL — the lognormal truncation
  window and the spacing convention.
* [YASF][yasf-url] — the spacing convention.
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
  — the layout of this README.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



[pypi-shield]: https://img.shields.io/pypi/v/polydisc.svg?style=for-the-badge
[pypi-url]: https://pypi.org/project/polydisc/
[tests-shield]: https://img.shields.io/github/actions/workflow/status/arunoruto/polydisc/test.yml?branch=main&style=for-the-badge&label=tests
[tests-url]: https://github.com/arunoruto/polydisc/actions/workflows/test.yml
[stars-shield]: https://img.shields.io/github/stars/arunoruto/polydisc.svg?style=for-the-badge
[stars-url]: https://github.com/arunoruto/polydisc/stargazers
[forks-shield]: https://img.shields.io/github/forks/arunoruto/polydisc.svg?style=for-the-badge
[forks-url]: https://github.com/arunoruto/polydisc/network/members
[issues-shield]: https://img.shields.io/github/issues/arunoruto/polydisc.svg?style=for-the-badge
[issues-url]: https://github.com/arunoruto/polydisc/issues
[license-shield]: https://img.shields.io/github/license/arunoruto/polydisc.svg?style=for-the-badge
[license-url]: https://github.com/arunoruto/polydisc/blob/main/LICENSE
[python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org
[numpy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org
[numba-shield]: https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white
[numba-url]: https://numba.pydata.org
[streamlit-shield]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io
[bridson-url]: https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
[pyfracval-url]: https://github.com/aetherspritee/PyFracVAL
[yasf-url]: https://github.com/AGBV/YASF
