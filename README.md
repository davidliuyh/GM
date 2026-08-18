# Gravitational Molecule

Numerical tools and notebooks for studying a gravitational molecule formed by a binary black hole and a bosonic cloud. The repository contains the original Mathematica calculations, a pure-Python reproduction of the v2 model, and a v3 extension for non-Hermitian cloud growth, decay, and signed mass exchange.

## Repository Layout

- `src/v2/gmlib.py` implements the pure-Python v2 numerical pipeline: hydrogenic basis states, operator integration, eigensystems, cloud moments, orbital evolution, gravitational-wave observables, detector quantities, and plotting helpers.
- `src/gmlib.py` re-exports the v2 API and adds the v3 three-point iteration model, tracked non-Hermitian eigensystems, biorthogonal observables, and mass exchange between the cloud and both black holes.
- `src/v2/gm.ipynb` validates the v2 calculation and demonstrates cache generation, orbital evolution, detection regions, and superradiant growth widths.
- `src/v2/plot.ipynb` reproduces the v2 plotting workflow up to the LISA noise-model section.
- `src/gm.ipynb` walks through one complete v3 cloud-termination step.
- `src/iteration.ipynb` evolves six independent tracked eigenstate branches with adaptive steps and resumable JSON checkpoints.
- `Mathematica/` contains the original v2 and v3 Wolfram notebooks used for comparison.
- `data/` contains optional NPZ caches and iteration checkpoints. See `data/README.md` for cache details.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib
- JupyterLab or Jupyter Notebook
- tqdm

Create an environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib jupyterlab tqdm
```

## Getting Started

Start Jupyter from the repository root:

```bash
jupyter lab
```

For the v2 reproduction, open `src/v2/gm.ipynb`. For a single v3 update, open `src/gm.ipynb`. For the full adaptive six-state calculation, open `src/iteration.ipynb`.

The notebooks locate the repository root automatically and add `src/` to `sys.path`. The Python modules can also be imported directly from a script launched with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python - <<'PY'
from gmlib import IntegrationSettings, IterationState, make_stencil

state = IterationState.from_ratios(
    primary_mass_solar=40.0,
    mass_ratio=0.99,
    alpha_primary=0.1,
    cloud_mass_fraction=0.01,
    separation=38.0,
)
print(make_stencil(state, delta_separation=0.1))
print(IntegrationSettings(workers=1, verbose=False))
PY
```

## Caches and Parallelism

`DataRepository` looks for matching `data/gm_tables_v1_*.npz` files before performing numerical integration. If a cache is missing and `compute_if_missing=True`, the v2 pipeline computes it and saves it atomically. Detection-grid caches use the `gm_detection_v1_*.npz` naming pattern.

Set `IntegrationSettings(workers=-1)` to use the available CPUs, subject to the implementation's worker cap. Cache generation is computationally expensive; preserving matching NPZ files makes notebook startup much faster.

The adaptive v3 notebook stores one atomic JSON checkpoint per eigenstate under `data/iteration_checkpoints/`. Its default `RESUME=True` setting resumes checkpoints only when their schema and configuration fingerprint match the current run.

## Numerical Conventions

- Arrays are separation-major: the leading axis indexes separation and the next axis indexes one of six eigenstates.
- v2 tracks Hermitian eigenstates by adjacent eigenvector overlap and Hungarian matching.
- v3 uses a three-point separation stencil, tracks complex non-Hermitian eigenstates, and evaluates cloud observables biorthogonally.
- Masses are exposed in solar masses where indicated; internal dynamical quantities use Planck units.

Implementation differences and known numerical caveats relative to Mathematica v2 are documented in `src/v2/diff.md`.
