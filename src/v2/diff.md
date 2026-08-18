# Mathematica v2 and Python v2 Differences

This note records differences between the Mathematica v2 and Python v2 implementations for future verification.

## Corresponding Implementations

- Mathematica v2: `Mathematica/v2/GMCalcLib_v2.0.nb`, `GMMain_v2.0.nb`, `GMMainPlot_v2.0.nb`, and their exported `.m` data.
- Python v2: `src/v2/gmlib.py` and the notebooks under `src/v2/`.

## Physical Definitions

Both implementations use the same six-state basis, Hamiltonian `H`, `dH/dOmega`, `omegaLz`, position and second-moment operators, GW frequency, mass quadrupole, and GR radiation-power combination. The corresponding Python GR expression retains `-16 Qxx (Qxy + Qyy)`. The cloud energy in the laboratory frame, corrected growth widths, derivative rules, and rounded constants also match Mathematica v2.

## Integration Implementation

Mathematica v2 combines analytic integration with `NIntegrate`; the numerical part applies `LocalAdaptive` separately to each eigenstate over `[-2R,2R]^3`.

Python v2 uses deterministic tensor-product Gauss-Legendre quadrature, partitioned near both cloud centers and the origin. Position and second-moment quantities are first constructed as reusable operator matrices and are explicitly Hermitianized. The formulas are therefore the same, but the numerical stability and error characteristics differ.

## Numerical Differences in the Base Matrices

Comparison conditions: `q=0.99`, 200 points, and `R ∈ [10,38]`.

| Quantity | Maximum absolute difference | RMS difference | Relative global scale |
|---|---:|---:|---:|
| `H` | 1.1478157645222059e-05 | 8.115145670131352e-07 | 3.881864931370566e-05 |
| `omegaLz` | 1.5628858649821553e-06 | 1.4291497713612603e-07 | 1.647561336951389e-05 |
| `dH/dOmega` | 1.9014375179382537e-04 | 1.2512458112510676e-05 | 5.949620445096506e-05 |

## Position Expectation Values

The maximum absolute difference in `Xc` is `1.9116056651851e-03`, and the RMS difference is `2.808246677041545e-04`.

For example, for state 1 at `R=38`, Python v2 gives `-18.904151922579`, while Mathematica v2 gives approximately `-18.90426392`, an absolute difference of about `1.12e-04`.

## Mathematica v2 Quadrupole Failure at Large R

At `R=38`, the six Python v2 `Qyy` values are

`[11.8324166254, 12.0686080382, 13.9459837059, 14.2342187741, 12.1553864820, 12.4082566624]`.

The values saved by Mathematica v2 are

`[5.2554e-13, 7.7919e-13, 2.2754e-12, 3.3120e-12, 5.3365e-13, 7.8994e-13]`.

The first location where the relative error exceeds 10% for each state is:

| State | Zero-based index | R |
|---:|---:|---:|
| 1 | 180 | 35.32663317 |
| 2 | 182 | 35.60804020 |
| 3 | 189 | 36.59296482 |
| 4 | 191 | 36.87437186 |
| 5 | 180 | 35.32663317 |
| 6 | 182 | 35.60804020 |

The source and saved data indicate that Mathematica's `LocalAdaptive` misses highly localized wave packets far from the origin as the cubic integration domain grows with R. The original `.m` files do not retain integration warnings, so this is a diagnosis based on the implementation and data behavior. `Qxx` and `Qzz` remain reasonable over the same region, but the incorrect `Qyy` propagates into the mass quadrupole and `PGR`. At large R, the Mathematica v2 `Qc` data should not be treated as a numerical reference.

## Eigenstate Labels

Mathematica v2 calls `Eigensystem` independently at each R. Its plotting code adjusts only phases and signs and does not globally track state permutations.

Python v2 tracks all six eigenstates using eigenvector overlaps at adjacent R points and Hungarian matching, then applies the same permutation to downstream physical quantities. Python v2 is therefore more explicit about preventing incorrect labels near crossings.

## Data Layout and Caching

- Mathematica v2 data is usually state-major and distributed across multiple `.m` files.
- Python v2 data is R-point-major, stored in one NPZ cache, and checked for shape, finiteness, and Hermiticity.

Axis order and state labels must be aligned before comparison; arrays cannot be subtracted directly using their original indices.

## Stencil Difference at R=38

The original grid saved by Mathematica v2 is

`xValues = Subdivide[10, 38, 199]`,

which contains 200 points and 199 intervals with spacing

`ΔR = 28/199 = 0.14070351758793898`.

Because `R=38` is the right endpoint of this grid, Mathematica v2 applies a first-order backward difference in `NumericalDerivatives`:

`F'(38) ≈ [F(38) - F(38-ΔR)]/ΔR`.

The extended Python v2 grid retains these 200 original points and evaluates one additional point at the right edge:

`38+ΔR = 38.14070351758794`.

The extended Python v2 grid therefore contains 201 points and 200 intervals. At `R=38`, its local three-point stencil is

`[37.85929648241206, 38, 38.14070351758794]`,

which gives the centered difference

`F'(38) ≈ [F(38+ΔR) - F(38-ΔR)]/(2ΔR)`.

This change does not affect `H`, `omegaLz`, `dH/dOmega`, `Xc`, or `Qc`, which are integrated directly on the original 200 bins. It changes only quantities that depend on R derivatives, including eigenenergy derivatives, cloud and system energy derivatives, GW frequencies, `PGR` derived from those frequencies, and the orbital evolution time.

Consequently, when comparing Mathematica v2 and the extended Python v2 implementation at `R=38`:

- Directly integrated quantities should be compared on the same original bins.
- If each implementation retains its own derivative rule, Mathematica uses a backward difference and Python uses a centered difference, so discrepancies are expected.
- To verify the formulas themselves, both implementations must use exactly the same stencil and difference formula.

## Evolution Solver

Mathematica v2 uses `Interpolation`, `NDSolve`, and `WhenEvent`; event-location convergence warnings appear in its saved output.

Python v2 uses `CubicSpline` and `solve_ivp`, explicitly detects turning points, and adds boundary guards and a Simpson estimate. The stopping events and interpolation errors are not identical, so numerical differences may appear near the end of orbital evolution.

## Default Parameters and Coverage

- Mathematica v2 plotting defaults: `M1=100`, `alpha=0.06`, `mc=0.1`, 200 points, and `R ∈ [10,38]`.
- Current Python v2 plotting defaults: `M1=100`, `alpha=0.1`, `mc=0.01`, 100 points, and `R ∈ [100,200]`.
- The Python v2 notebook covers only the Mathematica plotting workflow before the LISA noise section and does not reproduce all later plotting cells.

Masses, coupling, cloud mass, the R grid, state labels, and unit conventions must therefore be aligned explicitly before comparing outputs.
