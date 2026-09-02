# polydisc

Polydisperse Poisson-disc sampling for discs, spheres and flat carpets.

A Numba-accelerated, variable-radius variant of Bridson's algorithm
(*Fast Poisson disk sampling in arbitrary dimensions*, SIGGRAPH 2007,
algorithm 4.3) for 2D and 3D boxes: every particle gets its own
radius drawn from a distribution you choose, no two particles overlap,
and none pokes outside the box. Built for generating sphere carpets and
packings for light-scattering work, so it also tells you what it
*actually* produced, not just what you asked for.

Formerly `pysson-disc`.

## Install

```sh
uv sync            # or: devenv shell
```

Development uses [devenv](https://devenv.sh); `direnv allow` picks it up
automatically.

## Quickstart

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

## Flat carpets

A carpet is a packing in a slab thin enough that the spheres form a
layer. `carpet` wraps the 3D sampler; `monolayer_thickness` gives the
thinnest slab the radii allow.

```python
>>> spec = polydisc.RadiusSpec.lognormal(gmean=1.0, gstd=1.3)
>>> pack = polydisc.carpet(
...     (300.0, 300.0),
...     thickness=polydisc.monolayer_thickness(spec),
...     radii=spec,
...     seed=0,
... )
>>> pack.dimensions
3
>>> bool(pack.points[:, 2].max() <= pack.bounds[2])
True

```

```sh
polydisc carpet --extent 300,300 --dist lognormal --gmean 1 --gstd 1.3 \
    --seed 0 --out carpet.npz
```

Omitting `--thickness` uses the thinnest monolayer automatically.

Spheres pack in genuine 3D, so they may overlap seen from above;
`projected_area_fraction` is the summed disc area over the footprint,
which is the quantity single-scattering estimates want (and which can
exceed 1 for a thicker deposit).

## Spacing

Spacing is expressed as **how many mean radii of clearance sit between
neighbouring surfaces** — the same convention as
[PyFracVAL](https://github.com/aetherspritee/PyFracVAL) and YASF. It is
applied as a position-only dilation; radii are never touched.

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
by the same factor: the structure is stretched, not reshaped.

### Where does the extra room come from?

`fit_bounds` decides whether the domain or the particle count absorbs
the spacing.

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

* `fit_bounds=True` (default) samples a pre-shrunk box, then dilates, so
  the finished packing spans exactly the extent you asked for and holds
  fewer particles.
* `fit_bounds=False` samples the full box and lets the domain grow —
  PyFracVAL's behaviour, and the one that keeps the particle count.

For carpets, pass the same settings to `monolayer_thickness` so the
pre-shrink does not eat the slab:

```python
>>> thickness = polydisc.monolayer_thickness(spec, spacing=2.0)
>>> pack = polydisc.carpet(
...     (300.0, 300.0), thickness, radii=spec, spacing=2.0, seed=0
... )
>>> bool(pack.bounds[2] > 2 * spec.rmax)
True

```

## Radius distributions

| Constructor                                     | Parameters                       |
| ----------------------------------------------- | -------------------------------- |
| `RadiusSpec.discrete(values, weights=None)`     | a fixed set of radii             |
| `RadiusSpec.lognormal(gmean, gstd, truncate=2)` | geometric mean and spread        |
| `RadiusSpec.normal(mean, std, truncate=2)`      | arithmetic mean and spread       |
| `RadiusSpec.uniform(rmin, rmax)`                | flat over a range                |

Continuous families are truncated and sampled inside the kernel. The
lognormal defaults match FracVAL's `±2 GSD` window,
`[gmean / gstd², gmean · gstd²]`:

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

## Diagnostics: what you actually got

Two effects push the realised size distribution away from the request,
and both are reported rather than hidden.

**Boundary exclusion.** A particle of radius `r` can only be centred in
the shrunken box `b_d - 2r`, so big particles have fewer places to go.
Negligible in a roomy box, dominant in a thin slab — it is why a
"monolayer" of a wide distribution comes out almost entirely fine.
`geometric_bias` predicts it before you sample:

```python
>>> two_sizes = polydisc.RadiusSpec.discrete([1.0, 3.0])
>>> round(polydisc.geometric_bias(two_sizes, [200.0, 200.0])["bias"], 3)
-0.01
>>> round(polydisc.geometric_bias(two_sizes, [200.0, 200.0, 6.2])["bias"], 3)
-0.456

```

`sample` logs a warning when this exceeds 5%.

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
bounds            200 x 200
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

## Command line

```
polydisc generate   sample a 2D or 3D box
polydisc carpet     sample a flat carpet in a thin slab
polydisc report     print diagnostics for a saved .npz
polydisc radii      print a discrete radius set for pasting into a config
```

Both samplers accept the same options, from flags or a config file:

```sh
polydisc generate --bounds 500,500,50 --dist lognormal --gmean 2 --gstd 1.3 \
    --spacing 2 --repetitions 30 --seed 0 --out pack.npz
```

CSV goes to stdout when `--out` is omitted, with diagnostics on stderr
so a pipe stays clean:

```sh
polydisc carpet --extent 500,500 --radii 1,2 --seed 0 | head -5
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
extent = [500.0, 500.0]
spacing = 0.0

[profiles.sparse]
extent = [500.0, 500.0]
spacing = 3.0
```

```sh
polydisc carpet --config packing.toml --profile sparse --out sparse.npz
```

## Interactive inspector

```sh
streamlit run src/polydisc/app_streamlit.py
```

Sidebar controls for every sampling option, a 2D/3D view, the realised
radius histogram against the requested density, and the full report.

## Performance

Against the previous implementation, same seeds and same resulting
packings (`scripts/benchmark.py`):

| case                          |     N |    old |    new | speedup |
| ----------------------------- | ----: | -----: | -----: | ------: |
| 2D 1000², r=5                 |   6.0k |  0.044 s |  0.016 s |    2.8× |
| 2D 4000², r=1..5              | 729k |  8.63 s |  3.44 s |    2.5× |
| 2D 500², r=0.25..5 (20×)      |  57k |  2.20 s |  0.57 s |    3.9× |
| 3D 150³, r=1..5               |  59k |  6.77 s |  1.29 s |    5.2× |
| 3D 60³, r=0.5..5 (10×)        |  23k |  9.76 s |  1.75 s |    5.6× |
| 3D 300³, r=1..5               | 477k | 65.9 s | 15.6 s |    4.2× |

Peak memory on the last row fell from 698 MB to 176 MB.

Two changes account for most of it:

* **Cell size.** The acceleration grid uses `cell = 2 · rmax`, exactly
  the largest possible query radius, so the neighbour scan is always a
  3×3 (or 3×3×3) block *whatever* the `rmax / rmin` ratio. Sizing cells
  by `rmin` instead — the textbook choice, which guarantees at most one
  point per cell — makes the scan grow as `(rmax/rmin)^d` and the grid
  memory as `(1/rmin)^d`. Cells now hold several points in an intrusive
  linked list, so memory scales with the number of *particles*, not
  with the number of grid cells.
* **No allocation in the hot loop.** Candidate positions, directions and
  cell indices are all scalar arithmetic. A direct NumPy-style
  translation allocates several small arrays per candidate, which is
  where Numba spends its time.

## Notes

* Candidates are placed in the annulus `[r_i + r_j, 2(r_i + r_j)]` around
  the active point — Bridson's `[r, 2r]` generalised to two radii, with
  a uniform (not area-weighted) radial draw, which packs slightly denser.
* Seeding is done *inside* the kernel. Numba's generator is separate
  from NumPy's, so calling `np.random.seed` from Python does not reach
  it; pass `seed=` instead. Omitting it draws one from NumPy's global
  generator and records it on the result.
* The default `max_points` is the hard geometric ceiling
  `domain / smallest particle`, which cannot truncate a real run.

## Layout

```
src/polydisc/
    _kernels.py       Numba kernels: samplers, radius draws, gap search
    radii.py          RadiusSpec: discrete and continuous distributions
    sampling.py       sample(), Packing, geometric_bias()
    spacing.py        spacing -> scale factor
    carpet.py         thin-slab wrapper
    diagnostics.py    realised radii, gaps, report
    cli.py            polydisc command
    app_streamlit.py  interactive inspector
reference/            original MATLAB implementation
scripts/              example plots and the benchmark
```

## License

MIT
