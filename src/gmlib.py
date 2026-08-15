"""Single-step cloud-termination model for the gravitational molecule v3.

The v2 implementation remains available in :mod:`v2.gmlib`.  This module
re-exports that numerical API and adds the pieces needed by Mathematica/v3:
raw moment matrices, a tracked non-Hermitian eigensystem, biorthogonal cloud
observables, and an exactly integrated signed mass exchange with both black
holes.

Array convention
----------------
The leading axis of every iteration array is the three-point separation
stencil ``[R-dR, R, R+dR]``.  The second axis is the tracked eigenstate and the
last eigenvector axis follows the basis ordering
``BH1:{200,21-1,211}, BH2:{200,21-1,211}``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from math import factorial
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from v2 import gmlib as _v2
from v2.gmlib import *  # noqa: F401,F403 - retain the v2 public numerical API
from v2.gmlib import __all__ as _V2_ALL


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_BASIS_STATES: tuple[tuple[int, int, int], ...] = (
    (2, 0, 0),
    (2, 1, -1),
    (2, 1, 1),
    (2, 0, 0),
    (2, 1, -1),
    (2, 1, 1),
)


def _leading_derivative(values: ArrayLike, coordinates: ArrayLike) -> NDArray:
    """First/centered differences along axis zero, including complex data."""

    array = np.asarray(values)
    x = np.asarray(coordinates, dtype=np.float64)
    if array.shape[0] != x.size:
        raise ValueError("the leading values axis must match coordinates")
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinates must contain at least two increasing values")
    result = np.empty_like(array)
    result[0] = (array[1] - array[0]) / (x[1] - x[0])
    result[-1] = (array[-1] - array[-2]) / (x[-1] - x[-2])
    if x.size > 2:
        shape = (-1,) + (1,) * (array.ndim - 1)
        result[1:-1] = (array[2:] - array[:-2]) / (
            x[2:] - x[:-2]
        ).reshape(shape)
    return result


@dataclass(frozen=True)
class IterationState:
    """Physical state at the beginning or end of one orbital step.

    Black-hole and cloud masses are stored in solar masses.  The boson mass is
    fixed in Planck units, so ``alpha_primary`` and ``alpha_secondary`` change
    automatically after a signed mass exchange.
    """

    primary_mass_solar: float
    secondary_mass_solar: float
    cloud_mass_solar: float
    boson_mass: float
    spin_primary: float
    spin_secondary: float
    separation: float
    selected_state: int = 0

    def __post_init__(self) -> None:
        if self.primary_mass_solar <= 0.0 or self.secondary_mass_solar <= 0.0:
            raise ValueError("black-hole masses must be positive")
        if self.cloud_mass_solar < 0.0:
            raise ValueError("cloud mass cannot be negative")
        if self.boson_mass <= 0.0:
            raise ValueError("boson mass must be positive")
        if not (0.0 <= self.spin_primary <= 1.0):
            raise ValueError("spin_primary must lie in [0, 1]")
        if not (0.0 <= self.spin_secondary <= 1.0):
            raise ValueError("spin_secondary must lie in [0, 1]")
        if self.separation <= 0.0:
            raise ValueError("separation must be positive")
        if self.selected_state not in range(6):
            raise ValueError("selected_state must be in range(6)")

    @classmethod
    def from_ratios(
        cls,
        *,
        primary_mass_solar: float = 40.0,
        mass_ratio: float = 0.99,
        alpha_primary: float = 0.01,
        cloud_mass_fraction: float = 0.1,
        spin_primary: float = 1.0,
        spin_secondary: float = 1.0,
        separation: float = 38.0,
        selected_state: int = 0,
    ) -> "IterationState":
        if mass_ratio <= 0.0:
            raise ValueError("mass_ratio must be positive")
        if alpha_primary <= 0.0:
            raise ValueError("alpha_primary must be positive")
        if cloud_mass_fraction < 0.0:
            raise ValueError("cloud_mass_fraction cannot be negative")
        primary_planck = primary_mass_solar * _v2.SOLAR_MASS_PLANCK
        return cls(
            primary_mass_solar=float(primary_mass_solar),
            secondary_mass_solar=float(mass_ratio * primary_mass_solar),
            cloud_mass_solar=float(cloud_mass_fraction * primary_mass_solar),
            boson_mass=float(alpha_primary / primary_planck),
            spin_primary=float(spin_primary),
            spin_secondary=float(spin_secondary),
            separation=float(separation),
            selected_state=int(selected_state),
        )

    @property
    def primary_mass(self) -> float:
        return self.primary_mass_solar * _v2.SOLAR_MASS_PLANCK

    @property
    def secondary_mass(self) -> float:
        return self.secondary_mass_solar * _v2.SOLAR_MASS_PLANCK

    @property
    def cloud_mass(self) -> float:
        return self.cloud_mass_solar * _v2.SOLAR_MASS_PLANCK

    @property
    def mass_ratio(self) -> float:
        return self.secondary_mass_solar / self.primary_mass_solar

    @property
    def cloud_mass_fraction(self) -> float:
        return self.cloud_mass_solar / self.primary_mass_solar

    @property
    def alpha_primary(self) -> float:
        return self.boson_mass * self.primary_mass

    @property
    def alpha_secondary(self) -> float:
        return self.boson_mass * self.secondary_mass

    @property
    def energy_scale(self) -> float:
        return self.boson_mass * self.alpha_primary**2

    @property
    def bohr_radius_primary(self) -> float:
        return 1.0 / (self.boson_mass * self.alpha_primary)

    def summary(self) -> dict[str, float | int]:
        return {
            "M1_solar": self.primary_mass_solar,
            "M2_solar": self.secondary_mass_solar,
            "Mc_solar": self.cloud_mass_solar,
            "q": self.mass_ratio,
            "qc": self.cloud_mass_fraction,
            "alpha1": self.alpha_primary,
            "alpha2": self.alpha_secondary,
            "a1": self.spin_primary,
            "a2": self.spin_secondary,
            "R": self.separation,
            "state": self.selected_state + 1,
        }


@dataclass(frozen=True)
class IterationTables:
    """Raw three-point operators before choosing an eigensystem."""

    separation: FloatArray
    hamiltonian_dimless: FloatArray
    dh_domega: FloatArray
    omega_lz_dimless: FloatArray
    moments: ComplexArray

    def __post_init__(self) -> None:
        expected_matrix = (3, 6, 6)
        for name in ("hamiltonian_dimless", "dh_domega", "omega_lz_dimless"):
            if np.asarray(getattr(self, name)).shape != expected_matrix:
                raise ValueError(f"{name} must have shape {expected_matrix}")
        if np.asarray(self.separation).shape != (3,):
            raise ValueError("separation must have shape (3,)")
        if np.asarray(self.moments).shape != (3, 8, 6, 6):
            raise ValueError("moments must have shape (3, 8, 6, 6)")


@dataclass(frozen=True)
class TrackedEigensystem:
    """Continuously labelled right and biorthogonal left eigenvectors."""

    hamiltonian: ComplexArray
    growth_widths: FloatArray
    eigenvalues: ComplexArray
    right_vectors: ComplexArray
    left_bras: ComplexArray
    permutations: NDArray[np.int64]

    @property
    def biorthogonality(self) -> ComplexArray:
        return np.einsum(
            "nki,nji->nkj", self.left_bras, self.right_vectors, optimize=True
        )

    @property
    def eigen_residual_max(self) -> float:
        residual = np.einsum(
            "nij,nkj->nki", self.hamiltonian, self.right_vectors, optimize=True
        ) - self.eigenvalues[:, :, None] * self.right_vectors
        scale = max(float(np.max(np.abs(self.hamiltonian))), np.finfo(float).tiny)
        return float(np.max(np.abs(residual)) / scale)

    @property
    def biorthogonality_max_error(self) -> float:
        identity = np.broadcast_to(np.eye(6), (3, 6, 6))
        return float(np.max(np.abs(self.biorthogonality - identity)))


@dataclass(frozen=True)
class CloudObservables:
    """Complex biorthogonal diagnostics and their real conservative parts."""

    normalization: ComplexArray
    center_biorthogonal: ComplexArray
    quadrupole_biorthogonal: ComplexArray
    center: FloatArray
    quadrupole: FloatArray
    dh_domega_biorthogonal: ComplexArray
    cloud_energy_lab: ComplexArray
    cloud_energy_derivative: ComplexArray


@dataclass(frozen=True)
class OrbitalObservables:
    """Real conservative orbital quantities on the three-point stencil."""

    gw_frequency_squared: FloatArray
    mass_quadrupole: FloatArray
    gr_power: FloatArray
    system_energy: FloatArray
    system_energy_derivative: FloatArray
    delta_time: float


@dataclass(frozen=True)
class SignedRates:
    """Signed cloud growth rates for the selected tracked eigenstate."""

    channels: FloatArray
    black_hole_1: float
    black_hole_2: float
    total: float
    eigenvalue_total: float

    @property
    def consistency_error(self) -> float:
        return abs(self.total - self.eigenvalue_total)


@dataclass(frozen=True)
class MassExchange:
    """Exact constant-rate cloud/BH exchange over one orbital step."""

    delta_time: float
    exponent: float
    cloud_delta_solar: float
    primary_delta_solar: float
    secondary_delta_solar: float
    next_state: IterationState

    @property
    def conservation_error_solar(self) -> float:
        return abs(
            self.cloud_delta_solar
            + self.primary_delta_solar
            + self.secondary_delta_solar
        )


def make_stencil(state: IterationState, delta_separation: float) -> FloatArray:
    """Return the increasing stencil ``[R-dR, R, R+dR]``."""

    delta = float(delta_separation)
    if delta <= 0.0 or state.separation - delta <= 0.0:
        raise ValueError("delta_separation must be positive and smaller than R")
    return np.asarray(
        [state.separation - delta, state.separation, state.separation + delta],
        dtype=np.float64,
    )


def _compute_iteration_point(
    arguments: tuple[float, float, _v2.IntegrationSettings],
) -> tuple[FloatArray, FloatArray, FloatArray, ComplexArray]:
    mass_ratio, separation, settings = arguments
    hamiltonian, dh_domega, omega_lz = _v2._operator_matrices_at_separation(
        mass_ratio, separation, settings
    )
    moments = _v2._moment_matrices_at_separation(
        mass_ratio, separation, settings
    )
    return hamiltonian, dh_domega, omega_lz, moments


def compute_iteration_tables(
    state: IterationState,
    delta_separation: float,
    settings: _v2.IntegrationSettings | None = None,
) -> IterationTables:
    """Integrate all raw operators at ``R-dR``, ``R`` and ``R+dR``."""

    integration = settings or _v2.IntegrationSettings()
    separation = make_stencil(state, delta_separation)
    arguments = [
        (state.mass_ratio, float(radius), integration) for radius in separation
    ]
    workers = min(integration.resolved_workers, len(arguments))
    if workers == 1:
        results = list(map(_compute_iteration_point, arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_compute_iteration_point, arguments, chunksize=1))
    if integration.verbose:
        for index, radius in enumerate(separation, start=1):
            print(f"integrated stencil {index}/3: R={radius:.10g}")
    hamiltonian, dh_domega, omega_lz, moments = (
        np.stack(items) for items in zip(*results)
    )
    return IterationTables(
        separation=separation,
        hamiltonian_dimless=hamiltonian,
        dh_domega=dh_domega,
        omega_lz_dimless=omega_lz,
        moments=moments,
    )


def growth_width(
    basis_state: Sequence[int],
    spin: float,
    black_hole_mass: float,
    alpha: float,
    boson_mass: float,
) -> float:
    """Corrected growth width used by the Mathematica v3 notebook."""

    n, ell, m = (int(value) for value in basis_state)
    if not (0.0 <= spin <= 1.0):
        raise ValueError("dimensionless spin must lie in [0, 1]")
    r_plus = 1.0 + np.sqrt(1.0 - spin**2)
    omega_h = spin / (2.0 * black_hole_mass * r_plus)
    level_frequency = boson_mass * (1.0 - alpha**2 / (2.0 * n**2))
    coefficient = (
        2.0
        * r_plus
        * (2.0 ** (4 * ell + 1) * factorial(n + ell))
        / (n ** (2 * ell + 4) * factorial(n - ell - 1))
        * (
            factorial(ell)
            / (factorial(2 * ell) * factorial(2 * ell + 1))
        )
        ** 2
    )
    product = 1.0
    for k in range(1, ell + 1):
        product *= k**2 * (1.0 - spin**2) + (
            spin * m - 2.0 * r_plus * black_hole_mass * level_frequency
        ) ** 2
    return float(
        coefficient
        * product
        * (m * omega_h - level_frequency)
        * alpha ** (4 * ell + 5)
    )


def growth_widths(state: IterationState) -> FloatArray:
    """Return widths in the fixed six-component BH/basis ordering."""

    result = []
    for index, basis_state in enumerate(_BASIS_STATES):
        if index < 3:
            result.append(
                growth_width(
                    basis_state,
                    state.spin_primary,
                    state.primary_mass,
                    state.alpha_primary,
                    state.boson_mass,
                )
            )
        else:
            result.append(
                growth_width(
                    basis_state,
                    state.spin_secondary,
                    state.secondary_mass,
                    state.alpha_secondary,
                    state.boson_mass,
                )
            )
    return np.asarray(result, dtype=np.float64)


def build_nonhermitian_hamiltonians(
    state: IterationState,
    tables: IterationTables,
    *,
    include_growth: bool = True,
) -> tuple[ComplexArray, FloatArray]:
    """Add ``i Gamma`` to the dimensionful Hamiltonian diagonal."""

    hamiltonian = (
        tables.hamiltonian_dimless.astype(np.complex128) * state.energy_scale
    )
    widths = growth_widths(state) if include_growth else np.zeros(6)
    hamiltonian = hamiltonian.copy()
    diagonal = np.arange(6)
    hamiltonian[:, diagonal, diagonal] += 1j * widths
    return hamiltonian, widths


def track_eigensystem(
    eigenvalues: ArrayLike,
    eigenvectors: ArrayLike,
) -> tuple[ComplexArray, ComplexArray, NDArray[np.int64]]:
    """Match v2 Python's adjacent-overlap Hungarian state tracking."""

    tracked_values = np.asarray(eigenvalues, dtype=np.complex128).copy()
    tracked_vectors = np.asarray(eigenvectors, dtype=np.complex128).copy()
    if tracked_values.shape != (3, 6) or tracked_vectors.shape != (3, 6, 6):
        raise ValueError("expected eigenvalue/vector shapes (3,6) and (3,6,6)")
    permutations = np.empty((3, 6), dtype=np.int64)
    permutations[0] = np.arange(6)

    for point in range(1, 3):
        overlap = np.abs(
            tracked_vectors[point - 1].conj() @ tracked_vectors[point].T
        )
        previous_states, current_states = linear_sum_assignment(-overlap)
        permutation = current_states[np.argsort(previous_states)]
        permutations[point] = permutation
        tracked_values[point] = tracked_values[point, permutation]
        tracked_vectors[point] = tracked_vectors[point, permutation]

        adjacent_overlap = np.einsum(
            "ij,ij->i",
            tracked_vectors[point - 1].conj(),
            tracked_vectors[point],
        )
        phase = np.ones_like(adjacent_overlap)
        nonzero = np.abs(adjacent_overlap) > 0.0
        phase[nonzero] = np.exp(-1j * np.angle(adjacent_overlap[nonzero]))
        tracked_vectors[point] *= phase[:, None]
    return tracked_values, tracked_vectors, permutations


def diagonalize_and_track(
    hamiltonian: ArrayLike,
    widths: ArrayLike,
) -> TrackedEigensystem:
    """Diagonalize each point, sort by ``-|E|``, then track as in v2."""

    matrices = np.asarray(hamiltonian, dtype=np.complex128)
    if matrices.shape != (3, 6, 6):
        raise ValueError("hamiltonian must have shape (3, 6, 6)")
    values_at_points = []
    vectors_at_points = []
    for matrix in matrices:
        values, vector_columns = np.linalg.eig(matrix)
        order = np.argsort(-np.abs(values), kind="stable")
        values_at_points.append(values[order])
        vectors_at_points.append(vector_columns[:, order].T)
    values, vectors, permutations = track_eigensystem(
        np.asarray(values_at_points), np.asarray(vectors_at_points)
    )

    transpose_norm = np.einsum("nki,nki->nk", vectors, vectors, optimize=True)
    if np.any(np.abs(transpose_norm) < 1e-12):
        raise FloatingPointError("encountered a self-orthogonal exceptional point")
    left_bras = vectors / transpose_norm[:, :, None]
    return TrackedEigensystem(
        hamiltonian=matrices,
        growth_widths=np.asarray(widths, dtype=np.float64),
        eigenvalues=values,
        right_vectors=vectors,
        left_bras=left_bras,
        permutations=permutations,
    )


def biorthogonal_expectation(
    eigensystem: TrackedEigensystem,
    matrices: ArrayLike,
) -> ComplexArray:
    """Return ``<L|O|R>`` for all stencil points and tracked states."""

    operators = np.asarray(matrices)
    if operators.shape != (3, 6, 6):
        raise ValueError("matrices must have shape (3, 6, 6)")
    return np.einsum(
        "nki,nij,nkj->nk",
        eigensystem.left_bras,
        operators,
        eigensystem.right_vectors,
        optimize=True,
    )


def compute_cloud_observables(
    state: IterationState,
    tables: IterationTables,
    eigensystem: TrackedEigensystem,
) -> CloudObservables:
    """Compute v3's biorthogonal cloud center, quadrupole and lab energy."""

    moments = tables.moments
    normalization = biorthogonal_expectation(eigensystem, moments[:, 0])
    if np.any(np.abs(normalization) < 1e-12):
        raise FloatingPointError("cloud normalization is numerically zero")
    center_complex = (
        biorthogonal_expectation(eigensystem, moments[:, 1]) / normalization
    )
    quadrupole_complex = np.zeros((3, 6, 3, 3), dtype=np.complex128)
    moment_index = 2
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            values = (
                biorthogonal_expectation(
                    eigensystem, moments[:, moment_index]
                )
                / normalization
            )
            quadrupole_complex[:, :, first_axis, second_axis] = values
            quadrupole_complex[:, :, second_axis, first_axis] = values
            moment_index += 1

    dh_domega = biorthogonal_expectation(eigensystem, tables.dh_domega)
    omega_lz = biorthogonal_expectation(
        eigensystem, tables.omega_lz_dimless
    )
    cloud_energy = eigensystem.eigenvalues + omega_lz * state.energy_scale
    return CloudObservables(
        normalization=normalization,
        center_biorthogonal=center_complex,
        quadrupole_biorthogonal=quadrupole_complex,
        center=np.asarray(center_complex.real, dtype=np.float64),
        quadrupole=np.asarray(quadrupole_complex.real, dtype=np.float64),
        dh_domega_biorthogonal=dh_domega,
        cloud_energy_lab=cloud_energy,
        cloud_energy_derivative=_leading_derivative(
            cloud_energy, tables.separation
        ),
    )


def compute_gw_frequency_squared(
    state: IterationState,
    tables: IterationTables,
    eigensystem: TrackedEigensystem,
    cloud: CloudObservables,
) -> FloatArray:
    """Compute the real conservative GW frequency squared on the stencil."""

    p_mass = state.primary_mass
    s_mass = state.secondary_mass
    c_mass = state.cloud_mass
    total_mass = p_mass + s_mass + c_mass
    q = state.mass_ratio
    x = tables.separation[:, None]
    rb = state.bohr_radius_primary

    eigenvalue_derivative = _leading_derivative(
        eigensystem.eigenvalues, tables.separation
    ).real
    dh_expectation = cloud.dh_domega_biorthogonal.real
    dedr = eigenvalue_derivative + (
        1.5
        * state.energy_scale
        * np.sqrt(1.0 + q)
        * x ** (-1.5)
        * dh_expectation
        / x
    )

    system_center = c_mass * cloud.center / total_mass
    d1 = q / (1.0 + q) * x + system_center
    d2 = 1.0 / (1.0 + q) * x - system_center
    d1_derivative = _leading_derivative(d1, tables.separation)
    d2_derivative = _leading_derivative(d2, tables.separation)
    numerator = p_mass * s_mass / (x**2 * rb) + dedr * (
        c_mass / state.boson_mass
    )
    denominator = (
        p_mass * d1 * d1_derivative + s_mass * d2 * d2_derivative
    ) * rb**2
    frequency_squared = np.asarray(
        numerator / denominator / np.pi**2, dtype=np.float64
    )
    if not np.all(np.isfinite(frequency_squared)) or np.any(
        frequency_squared <= 0.0
    ):
        raise FloatingPointError("computed non-positive GW frequency squared")
    return frequency_squared


def compute_mass_quadrupole(
    state: IterationState,
    tables: IterationTables,
    cloud: CloudObservables,
) -> FloatArray:
    """Combine the real cloud quadrupole with the binary point masses."""

    p_mass = state.primary_mass
    s_mass = state.secondary_mass
    c_mass = state.cloud_mass
    total_mass = p_mass + s_mass + c_mass
    q = state.mass_ratio
    x = tables.separation[:, None]
    rb = state.bohr_radius_primary
    system_center = c_mass * cloud.center / total_mass

    q_bbh = (
        p_mass * (q / (1.0 + q) * x * rb + system_center * rb) ** 2
        + s_mass
        * (1.0 / (1.0 + q) * x * rb - system_center * rb) ** 2
    )
    mass_quadrupole = c_mass * cloud.quadrupole * rb**2
    mass_quadrupole = mass_quadrupole.copy()
    mass_quadrupole[:, :, 0, 0] += q_bbh
    return mass_quadrupole


def compute_gr_power(
    gw_frequency_squared: ArrayLike,
    mass_quadrupole: ArrayLike,
) -> FloatArray:
    """Compute quadrupole GW power for every stencil point and state."""

    frequency_squared = np.asarray(gw_frequency_squared, dtype=np.float64)
    quad = np.asarray(mass_quadrupole, dtype=np.float64)
    if frequency_squared.shape != (3, 6) or quad.shape != (3, 6, 3, 3):
        raise ValueError("unexpected frequency or mass-quadrupole shape")
    bracket = (
        16.0 * quad[:, :, 0, 0] ** 2
        + 72.0 * quad[:, :, 0, 1] ** 2
        + quad[:, :, 0, 2] ** 2
        + 8.0 * quad[:, :, 1, 1] ** 2
        - 16.0
        * quad[:, :, 0, 0]
        * (quad[:, :, 0, 1] + quad[:, :, 1, 1])
        + quad[:, :, 1, 2] ** 2
    )
    gr_power = np.asarray(
        (2.0 / 5.0) * bracket * (np.pi**2 * frequency_squared) ** 3,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(gr_power)) or np.any(gr_power <= 0.0):
        raise FloatingPointError("computed non-positive GR power")
    return gr_power


def compute_system_energy(
    state: IterationState,
    tables: IterationTables,
    cloud: CloudObservables,
    gw_frequency_squared: ArrayLike,
) -> FloatArray:
    """Return the real conservative cloud+binary energy on the stencil."""

    p_mass = state.primary_mass
    s_mass = state.secondary_mass
    c_mass = state.cloud_mass
    total_mass = p_mass + s_mass + c_mass
    q = state.mass_ratio
    x = tables.separation[:, None]
    rb = state.bohr_radius_primary
    frequency_squared = np.asarray(gw_frequency_squared, dtype=np.float64)
    if frequency_squared.shape != (3, 6):
        raise ValueError("gw_frequency_squared must have shape (3, 6)")

    rho1 = q / (1.0 + q) * x
    rho2 = 1.0 / (1.0 + q) * x
    center_of_mass = (
        c_mass * cloud.center - p_mass * rho1 + s_mass * rho2
    ) / total_mass
    energy_d1 = q / (1.0 + q) * x + center_of_mass
    energy_d2 = 1.0 / (1.0 + q) * x - center_of_mass
    system_energy = (
        (c_mass / state.boson_mass) * cloud.cloud_energy_lab.real
        - p_mass * s_mass / (x * rb)
        + 0.5
        * p_mass
        * (np.pi**2 * frequency_squared)
        * (energy_d1 * rb) ** 2
        + 0.5
        * s_mass
        * (np.pi**2 * frequency_squared)
        * (energy_d2 * rb) ** 2
    )
    return np.asarray(system_energy, dtype=np.float64)


def compute_step_duration(
    state: IterationState,
    delta_separation: float,
    tables: IterationTables,
    system_energy: ArrayLike,
    gr_power: ArrayLike,
) -> tuple[FloatArray, float]:
    """Apply ``dE/dR dR/dt = -P`` at the central stencil point."""

    energy = np.asarray(system_energy, dtype=np.float64)
    power = np.asarray(gr_power, dtype=np.float64)
    if energy.shape != (3, 6) or power.shape != (3, 6):
        raise ValueError("system_energy and gr_power must have shape (3, 6)")
    system_energy_derivative = np.asarray(
        _leading_derivative(energy, tables.separation),
        dtype=np.float64,
    )
    selected = state.selected_state
    derivative_at_center = float(system_energy_derivative[1, selected])
    power_at_center = float(gr_power[1, selected])
    delta_time = float(delta_separation * derivative_at_center / power_at_center)
    if not np.isfinite(delta_time) or delta_time <= 0.0:
        raise FloatingPointError(
            f"invalid orbital step duration {delta_time}; expected dE/dR > 0"
        )
    return system_energy_derivative, delta_time


def compute_orbital_observables(
    state: IterationState,
    delta_separation: float,
    tables: IterationTables,
    eigensystem: TrackedEigensystem,
    cloud: CloudObservables,
) -> OrbitalObservables:
    """Convenience composition of the separately inspectable orbital steps."""

    frequency_squared = compute_gw_frequency_squared(
        state, tables, eigensystem, cloud
    )
    mass_quadrupole = compute_mass_quadrupole(state, tables, cloud)
    gr_power = compute_gr_power(frequency_squared, mass_quadrupole)
    system_energy = compute_system_energy(
        state, tables, cloud, frequency_squared
    )
    system_energy_derivative, delta_time = compute_step_duration(
        state,
        delta_separation,
        tables,
        system_energy,
        gr_power,
    )
    return OrbitalObservables(
        gw_frequency_squared=frequency_squared,
        mass_quadrupole=mass_quadrupole,
        gr_power=gr_power,
        system_energy=system_energy,
        system_energy_derivative=system_energy_derivative,
        delta_time=delta_time,
    )


def compute_signed_rates(
    state: IterationState,
    eigensystem: TrackedEigensystem,
) -> SignedRates:
    """Split ``2 Im(E)`` into signed BH1/BH2 basis-channel rates."""

    vector = eigensystem.right_vectors[1, state.selected_state]
    probability = np.abs(vector) ** 2
    probability /= np.sum(probability)
    channels = 2.0 * eigensystem.growth_widths * probability
    first = float(np.sum(channels[:3]))
    second = float(np.sum(channels[3:]))
    total = first + second
    eigenvalue_total = float(
        2.0 * eigensystem.eigenvalues[1, state.selected_state].imag
    )
    scale = max(abs(eigenvalue_total), np.finfo(float).tiny)
    if abs(total - eigenvalue_total) > 2e-7 * scale:
        raise FloatingPointError(
            "basis-channel rates do not reproduce the eigenvalue imaginary part"
        )
    return SignedRates(
        channels=np.asarray(channels, dtype=np.float64),
        black_hole_1=first,
        black_hole_2=second,
        total=total,
        eigenvalue_total=eigenvalue_total,
    )


def apply_mass_exchange(
    state: IterationState,
    delta_separation: float,
    delta_time: float,
    rates: SignedRates,
) -> MassExchange:
    """Exactly integrate constant signed rates and construct the next state."""

    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    exponent = float(rates.total * delta_time)
    cloud_delta = float(state.cloud_mass_solar * np.expm1(exponent))
    if rates.total == 0.0:
        integrated_cloud = state.cloud_mass_solar * delta_time
    else:
        integrated_cloud = cloud_delta / rates.total
    primary_delta = float(-rates.black_hole_1 * integrated_cloud)
    secondary_delta = float(-rates.black_hole_2 * integrated_cloud)
    next_state = replace(
        state,
        primary_mass_solar=state.primary_mass_solar + primary_delta,
        secondary_mass_solar=state.secondary_mass_solar + secondary_delta,
        cloud_mass_solar=state.cloud_mass_solar + cloud_delta,
        separation=state.separation - float(delta_separation),
    )
    exchange = MassExchange(
        delta_time=float(delta_time),
        exponent=exponent,
        cloud_delta_solar=cloud_delta,
        primary_delta_solar=primary_delta,
        secondary_delta_solar=secondary_delta,
        next_state=next_state,
    )
    reference_mass = (
        state.primary_mass_solar
        + state.secondary_mass_solar
        + state.cloud_mass_solar
    )
    if exchange.conservation_error_solar > 2e-14 * reference_mass:
        raise FloatingPointError("cloud/BH mass exchange failed conservation")
    return exchange


__all__ = [
    *_V2_ALL,
    "CloudObservables",
    "IterationState",
    "IterationTables",
    "MassExchange",
    "OrbitalObservables",
    "SignedRates",
    "TrackedEigensystem",
    "apply_mass_exchange",
    "biorthogonal_expectation",
    "build_nonhermitian_hamiltonians",
    "compute_cloud_observables",
    "compute_gr_power",
    "compute_gw_frequency_squared",
    "compute_iteration_tables",
    "compute_mass_quadrupole",
    "compute_orbital_observables",
    "compute_signed_rates",
    "compute_step_duration",
    "compute_system_energy",
    "diagonalize_and_track",
    "growth_width",
    "growth_widths",
    "make_stencil",
    "track_eigensystem",
]
