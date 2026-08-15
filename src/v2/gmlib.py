"""Pure-Python numerical tools for the v2 gravitational-molecule (GM) model.

The module translates the complete numerical path of the original notebooks:
hydrogenic basis states, reduced-Hamiltonian/operator integrals, cloud moments,
eigensystems, orbital dynamics, radiation power, detector quantities, and
figures.  It does not require Wolfram/Mathematica or Wolfram ``.m`` files.

The public API deliberately avoids Mathematica-style global state:

``DataRepository``
    Loads standalone ``data/*.npz`` caches and computes missing model-table
    caches with the pure-Python integration pipeline.
``GMParameters``
    Holds the physical parameters and derived Planck-unit scales.
``GMModel``
    Computes all model-dependent arrays and inspiral summaries lazily.

Array convention
----------------
Python results use separation as the first axis and eigenstate as the second
axis, i.e. ``(n_separations, 6)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from math import factorial
import os
from pathlib import Path
from typing import Callable, Final, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad, simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# The original notebooks use these rounded constants.  Keeping them here is
# important for numerical agreement with the saved outputs.
PLANCK_TIME_S: Final = 5.39e-44
PLANCK_LENGTH_M: Final = 1.616e-35
PLANCK_MASS_KG: Final = 2.176e-8
SOLAR_MASS_KG: Final = 1.989e30
SOLAR_MASS_PLANCK: Final = SOLAR_MASS_KG / PLANCK_MASS_KG
SECONDS_PER_YEAR: Final = 365.0 * 24.0 * 3600.0
PLANCK_TIMES_PER_YEAR: Final = SECONDS_PER_YEAR / PLANCK_TIME_S

SPEED_OF_LIGHT: Final = 299_792_458.0
NEWTON_G: Final = 6.67e-11
MPC_M: Final = 3.086e22
HUBBLE_KM_S_MPC: Final = 67.4e3
OMEGA_M: Final = 0.315
OMEGA_LAMBDA: Final = 0.685


def numerical_derivative(values: ArrayLike, coordinates: ArrayLike) -> FloatArray:
    """Reproduce ``NumericalDerivatives`` from ``GMCalcLib_v2.0.nb``.

    End points use one-sided first differences and interior points use the
    centered secant through their two neighbours.  The leading axis is the
    coordinate axis.
    """

    array = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    if array.shape[0] != x.size:
        raise ValueError("the leading values axis must match coordinates")
    if x.size < 2:
        raise ValueError("at least two coordinates are required")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")

    derivative = np.empty_like(array)
    reshape = (-1,) + (1,) * (array.ndim - 1)
    derivative[0] = (array[1] - array[0]) / (x[1] - x[0])
    derivative[-1] = (array[-1] - array[-2]) / (x[-1] - x[-2])
    if x.size > 2:
        width = (x[2:] - x[:-2]).reshape(reshape)
        derivative[1:-1] = (array[2:] - array[:-2]) / width
    return derivative


@dataclass(frozen=True)
class IntegrationSettings:
    """Accuracy and parallelism controls for cache-miss integrations.

    The domain partition mirrors the singularity-aware intervals used in the
    original numerical integration.  Each smooth subinterval is integrated by
    deterministic Gauss-Legendre quadrature.  ``operator_order=32`` and
    ``moment_order=16`` reproduce the reference operators to roughly ``1e-5``
    absolute accuracy while remaining practical on a parallel CPU node.
    """

    operator_order: int = 32
    moment_order: int = 16
    operator_limit: float = 55.0
    moment_limit_factor: float = 2.0
    exclusion_delta: float = 0.01
    workers: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.operator_order < 4 or self.moment_order < 4:
            raise ValueError("quadrature orders must be at least four")
        if self.operator_limit <= 0.0 or self.moment_limit_factor <= 0.0:
            raise ValueError("integration limits must be positive")
        if self.exclusion_delta <= 0.0:
            raise ValueError("exclusion_delta must be positive")
        if self.workers == 0 or self.workers < -1:
            raise ValueError("workers must be -1 or a positive integer")

    @property
    def resolved_workers(self) -> int:
        # NumPy's BLAS already uses several threads inside each worker.  A cap
        # avoids severe nested-thread oversubscription on 256-core CPU nodes.
        return min(32, max(1, os.cpu_count() or 1)) if self.workers == -1 else self.workers


def _orbital_frequency_dimless(mass_ratio: float, separation: float) -> float:
    return np.sqrt(1.0 + mass_ratio) / separation**1.5


def _basis_states(
    points: FloatArray,
    mass_ratio: float,
    separation: float,
    center: float,
    primary: bool,
    omega_dimless: float,
    *,
    derivatives: bool = False,
) -> ComplexArray | tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Dimensionless 200, 21-1, and 211 hydrogenic basis states.

    Coordinates are measured in the primary cloud's Bohr radius.  The boost
    phases place both clouds in the co-rotating frame.  When requested, the
    final two arrays are analytic x/y derivatives used by ``omega Lz``.
    """

    q = mass_ratio
    x, y, z = points.T
    dx = x - center
    radius = np.sqrt(dx * dx + y * y + z * z)
    inverse_radius = np.divide(
        1.0, radius, out=np.zeros_like(radius), where=radius != 0.0
    )
    if primary:
        radial_scale = 1.0
        amplitude_200 = 1.0 / (4.0 * np.sqrt(2.0 * np.pi))
        amplitude_21 = 1.0 / (8.0 * np.sqrt(np.pi))
        phase_wave_number = -omega_dimless * q * separation / (1.0 + q)
    else:
        radial_scale = q
        amplitude_200 = q**1.5 / (4.0 * np.sqrt(2.0 * np.pi))
        amplitude_21 = q**2.5 / (8.0 * np.sqrt(np.pi))
        phase_wave_number = omega_dimless * separation / (1.0 + q)

    decay = np.exp(-0.5 * radial_scale * radius)
    phase = np.exp(1j * phase_wave_number * y)
    radial_200 = (2.0 - radial_scale * radius) * decay
    polynomial_minus = dx - 1j * y
    polynomial_plus = -(dx + 1j * y)
    states = np.stack(
        [
            amplitude_200 * radial_200 * phase,
            amplitude_21 * polynomial_minus * decay * phase,
            amplitude_21 * polynomial_plus * decay * phase,
        ],
        axis=1,
    )
    if not derivatives:
        return states

    radial_200_derivative = (
        0.5 * radial_scale**2 * radius - 2.0 * radial_scale
    ) * decay
    derivative_x = np.empty_like(states)
    derivative_y = np.empty_like(states)
    derivative_x[:, 0] = (
        amplitude_200 * radial_200_derivative * dx * inverse_radius * phase
    )
    derivative_y[:, 0] = amplitude_200 * (
        radial_200_derivative * y * inverse_radius
        + 1j * phase_wave_number * radial_200
    ) * phase

    polynomials = (polynomial_minus, polynomial_plus)
    polynomial_x = (np.ones_like(x), -np.ones_like(x))
    polynomial_y = (-1j * np.ones_like(y), -1j * np.ones_like(y))
    for state_index, (polynomial, derivative_polynomial_x, derivative_polynomial_y) in enumerate(
        zip(polynomials, polynomial_x, polynomial_y), start=1
    ):
        derivative_x[:, state_index] = amplitude_21 * (
            derivative_polynomial_x
            - 0.5 * radial_scale * polynomial * dx * inverse_radius
        ) * decay * phase
        derivative_y[:, state_index] = amplitude_21 * (
            derivative_polynomial_y
            - 0.5 * radial_scale * polynomial * y * inverse_radius
            + 1j * phase_wave_number * polynomial
        ) * decay * phase
    return states, derivative_x, derivative_y


def hydrogenic_basis_states(
    points: ArrayLike,
    mass_ratio: float,
    separation: float,
) -> ComplexArray:
    """Return all six co-rotating basis states at Cartesian ``points``.

    This public helper is useful for inspecting cloud densities independently
    of the cache-generation pipeline.
    """

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape (n, 3) or (3,)")
    q = float(mass_ratio)
    radius = float(separation)
    omega = _orbital_frequency_dimless(q, radius)
    rho_primary = q * radius / (1.0 + q)
    rho_secondary = radius / (1.0 + q)
    return np.concatenate(
        [
            _basis_states(
                coordinates, q, radius, -rho_primary, True, omega
            ),
            _basis_states(
                coordinates, q, radius, rho_secondary, False, omega
            ),
        ],
        axis=1,
    )


def _operator_integrand(
    points: FloatArray,
    mass_ratio: float,
    separation: float,
) -> ComplexArray:
    """Return H, dH/dOmega, and omega-Lz integrands for all matrix entries."""

    q = mass_ratio
    radius = separation
    omega = _orbital_frequency_dimless(q, radius)
    rho_primary = q * radius / (1.0 + q)
    rho_secondary = radius / (1.0 + q)
    primary, primary_dx, primary_dy = _basis_states(
        points, q, radius, 0.0, True, omega, derivatives=True
    )
    secondary_at_primary = _basis_states(
        points, q, radius, radius, False, omega
    )
    primary_at_secondary = _basis_states(
        points, q, radius, -radius, True, omega
    )
    secondary, secondary_dx, secondary_dy = _basis_states(
        points, q, radius, 0.0, False, omega, derivatives=True
    )
    x, y, z = points.T
    distance_secondary = np.sqrt((x - radius) ** 2 + y**2 + z**2)
    distance_primary = np.sqrt((x + radius) ** 2 + y**2 + z**2)
    distance_secondary = np.maximum(distance_secondary, np.finfo(float).tiny)
    distance_primary = np.maximum(distance_primary, np.finfo(float).tiny)
    magnetic_sign = np.asarray([0.0, 1.0, -1.0])

    primary_factor = (
        -0.125
        + magnetic_sign[None, :] * omega
        - q / distance_secondary[:, None]
        + omega**2
        * rho_primary
        * (x[:, None] - 0.5 * rho_primary)
    )
    secondary_factor = (
        -0.125 * q**2
        + magnetic_sign[None, :] * omega
        - 1.0 / distance_primary[:, None]
        + omega**2
        * rho_secondary
        * (-x[:, None] - 0.5 * rho_secondary)
    )
    bra_primary_frame = np.concatenate([primary, secondary_at_primary], axis=1)
    bra_secondary_frame = np.concatenate([primary_at_secondary, secondary], axis=1)

    hamiltonian = np.empty((points.shape[0], 6, 6), dtype=np.complex128)
    hamiltonian[:, :, :3] = (
        np.conj(bra_primary_frame)[:, :, None]
        * (primary * primary_factor)[:, None, :]
    )
    hamiltonian[:, :, 3:] = (
        np.conj(bra_secondary_frame)[:, :, None]
        * (secondary * secondary_factor)[:, None, :]
    )

    primary_derivative_omega = -1j * rho_primary * y[:, None] * primary
    secondary_derivative_omega = 1j * rho_secondary * y[:, None] * secondary
    bra_primary_derivative = np.concatenate(
        [
            primary_derivative_omega,
            1j * rho_secondary * y[:, None] * secondary_at_primary,
        ],
        axis=1,
    )
    bra_secondary_derivative = np.concatenate(
        [
            -1j * rho_primary * y[:, None] * primary_at_secondary,
            secondary_derivative_omega,
        ],
        axis=1,
    )
    primary_factor_derivative = (
        magnetic_sign[None, :]
        + 2.0
        * omega
        * rho_primary
        * (x[:, None] - 0.5 * rho_primary)
    )
    secondary_factor_derivative = (
        magnetic_sign[None, :]
        + 2.0
        * omega
        * rho_secondary
        * (-x[:, None] - 0.5 * rho_secondary)
    )
    primary_h_derivative = (
        primary_derivative_omega * primary_factor
        + primary * primary_factor_derivative
    )
    secondary_h_derivative = (
        secondary_derivative_omega * secondary_factor
        + secondary * secondary_factor_derivative
    )
    dh_domega = np.empty_like(hamiltonian)
    dh_domega[:, :, :3] = (
        np.conj(bra_primary_derivative)[:, :, None]
        * (primary * primary_factor)[:, None, :]
        + np.conj(bra_primary_frame)[:, :, None]
        * primary_h_derivative[:, None, :]
    )
    dh_domega[:, :, 3:] = (
        np.conj(bra_secondary_derivative)[:, :, None]
        * (secondary * secondary_factor)[:, None, :]
        + np.conj(bra_secondary_frame)[:, :, None]
        * secondary_h_derivative[:, None, :]
    )

    primary_lz = omega * (
        -1j * (x[:, None] * primary_dy - y[:, None] * primary_dx)
        + 1j * rho_primary * primary_dy
    )
    secondary_lz = omega * (
        -1j * (x[:, None] * secondary_dy - y[:, None] * secondary_dx)
        - 1j * rho_secondary * secondary_dy
    )
    omega_lz = np.empty_like(hamiltonian)
    omega_lz[:, :, :3] = (
        np.conj(bra_primary_frame)[:, :, None] * primary_lz[:, None, :]
    )
    omega_lz[:, :, 3:] = (
        np.conj(bra_secondary_frame)[:, :, None] * secondary_lz[:, None, :]
    )
    return np.stack([hamiltonian, dh_domega, omega_lz], axis=1)


def _moment_integrand(
    points: FloatArray,
    mass_ratio: float,
    separation: float,
) -> ComplexArray:
    basis = hydrogenic_basis_states(points, mass_ratio, separation)
    overlaps = np.conj(basis)[:, :, None] * basis[:, None, :]
    values = [overlaps, points[:, 0, None, None] * overlaps]
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            values.append(
                points[:, first_axis, None, None]
                * points[:, second_axis, None, None]
                * overlaps
            )
    return np.stack(values, axis=1)


def _partition_edges(
    limit: float,
    centers: Sequence[float],
    delta: float,
) -> FloatArray:
    edges = [-limit, limit]
    for center in centers:
        if -limit < center < limit:
            edges.extend([max(-limit, center - delta), min(limit, center + delta)])
    result = np.unique(np.asarray(edges, dtype=np.float64))
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("invalid integration partition")
    return result


def _composite_gauss_legendre(edges: FloatArray, order: int) -> tuple[FloatArray, FloatArray]:
    base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
    nodes: list[FloatArray] = []
    weights: list[FloatArray] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (upper - lower) * base_nodes + 0.5 * (upper + lower))
        weights.append(0.5 * (upper - lower) * base_weights)
    return np.concatenate(nodes), np.concatenate(weights)


def _tensor_product_integral(
    integrand: Callable[[FloatArray, float, float], ComplexArray],
    mass_ratio: float,
    separation: float,
    order: int,
    x_edges: FloatArray,
    transverse_edges: FloatArray,
) -> ComplexArray:
    x_nodes, x_weights = _composite_gauss_legendre(x_edges, order)
    y_nodes, y_weights = _composite_gauss_legendre(transverse_edges, order)
    z_nodes, z_weights = _composite_gauss_legendre(transverse_edges, order)
    y_grid, z_grid = np.meshgrid(y_nodes, z_nodes, indexing="ij")
    transverse_weights = (y_weights[:, None] * z_weights[None, :]).ravel()
    total: ComplexArray | None = None
    for x_value, x_weight in zip(x_nodes, x_weights):
        points = np.column_stack(
            [
                np.full(y_grid.size, x_value),
                y_grid.ravel(),
                z_grid.ravel(),
            ]
        )
        values = integrand(points, mass_ratio, separation)
        current = x_weight * np.einsum(
            "n,n...->...", transverse_weights, values, optimize=True
        )
        total = current if total is None else total + current
    if total is None:
        raise RuntimeError("empty integration grid")
    return total


def _hermitian_from_upper(matrix: ComplexArray) -> FloatArray:
    upper = np.triu(matrix.real)
    return upper + np.triu(upper, 1).T


def _operator_matrices_at_separation(
    mass_ratio: float,
    separation: float,
    settings: IntegrationSettings,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    limit = settings.operator_limit
    x_edges = _partition_edges(
        limit, (-separation, 0.0, separation), settings.exclusion_delta
    )
    transverse_edges = _partition_edges(
        limit, (0.0,), settings.exclusion_delta
    )
    matrices = _tensor_product_integral(
        _operator_integrand,
        mass_ratio,
        separation,
        settings.operator_order,
        x_edges,
        transverse_edges,
    )
    return tuple(_hermitian_from_upper(matrix) for matrix in matrices)  # type: ignore[return-value]


def _moment_matrices_at_separation(
    mass_ratio: float,
    separation: float,
    settings: IntegrationSettings,
) -> ComplexArray:
    limit = settings.moment_limit_factor * separation
    rho_primary = mass_ratio * separation / (1.0 + mass_ratio)
    rho_secondary = separation / (1.0 + mass_ratio)
    x_edges = _partition_edges(
        limit,
        (-rho_primary, 0.0, rho_secondary),
        settings.exclusion_delta,
    )
    transverse_edges = _partition_edges(
        limit, (0.0,), settings.exclusion_delta
    )
    moments = _tensor_product_integral(
        _moment_integrand,
        mass_ratio,
        separation,
        settings.moment_order,
        x_edges,
        transverse_edges,
    )
    return 0.5 * (moments + np.swapaxes(moments.conj(), -1, -2))


def _cloud_moments_from_matrices(
    eigenvectors: ComplexArray,
    moments: ComplexArray,
) -> tuple[FloatArray, FloatArray]:
    total_mass = np.einsum(
        "ki,ij,kj->k", eigenvectors.conj(), moments[0], eigenvectors, optimize=True
    ).real
    center = (
        np.einsum(
            "ki,ij,kj->k",
            eigenvectors.conj(),
            moments[1],
            eigenvectors,
            optimize=True,
        ).real
        / total_mass
    )
    quadrupole = np.zeros((6, 3, 3), dtype=np.float64)
    moment_index = 2
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            values = (
                np.einsum(
                    "ki,ij,kj->k",
                    eigenvectors.conj(),
                    moments[moment_index],
                    eigenvectors,
                    optimize=True,
                ).real
                / total_mass
            )
            quadrupole[:, first_axis, second_axis] = values
            quadrupole[:, second_axis, first_axis] = values
            moment_index += 1
    return center, quadrupole


def _compute_separation_tables(
    arguments: tuple[float, float, IntegrationSettings],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    mass_ratio, separation, settings = arguments
    hamiltonian, dh_domega, omega_lz = _operator_matrices_at_separation(
        mass_ratio, separation, settings
    )
    _, eigenvector_columns = np.linalg.eigh(hamiltonian)
    eigenvectors = eigenvector_columns.T.astype(np.complex128)
    moments = _moment_matrices_at_separation(mass_ratio, separation, settings)
    center, quadrupole = _cloud_moments_from_matrices(eigenvectors, moments)
    return hamiltonian, omega_lz, dh_domega, center, quadrupole


def compute_model_tables(
    mass_ratio: float,
    separations: ArrayLike,
    settings: IntegrationSettings | None = None,
) -> "ModelTables":
    """Numerically generate all integral tables without external software."""

    q = float(mass_ratio)
    if q <= 0.0:
        raise ValueError("mass_ratio must be positive")
    x = np.asarray(separations, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or np.any(x <= 0.0):
        raise ValueError("separations must be a non-empty positive 1D array")
    integration = settings or IntegrationSettings()
    arguments = [(q, float(value), integration) for value in x]
    workers = min(integration.resolved_workers, x.size)
    if workers == 1:
        iterator = map(_compute_separation_tables, arguments)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_compute_separation_tables, arguments, chunksize=1)

    results = []
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if integration.verbose:
                print(f"integrated separation {index}/{x.size}: R={x[index - 1]:.8g}")
    finally:
        if pool is not None:
            pool.shutdown()
    hamiltonian, omega_lz, dh_domega, center, quadrupole = (
        np.stack(values) for values in zip(*results)
    )
    return ModelTables(
        separation=x,
        hamiltonian_dimless=hamiltonian,
        omega_lz_dimless=omega_lz,
        dh_domega=dh_domega,
        cloud_center=center,
        cloud_quadrupole=quadrupole,
    )


@dataclass(frozen=True)
class GMParameters:
    """Physical parameters in the convention of ``GMInitialize``."""

    mass_ratio: float = 0.99
    primary_mass_solar: float = 40.0
    alpha: float = 0.01
    cloud_mass_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.mass_ratio <= 0.0:
            raise ValueError("mass_ratio must be positive")
        if self.primary_mass_solar <= 0.0:
            raise ValueError("primary_mass_solar must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.cloud_mass_fraction < 0.0:
            raise ValueError("cloud_mass_fraction cannot be negative")

    @property
    def primary_mass(self) -> float:
        return self.primary_mass_solar * SOLAR_MASS_PLANCK

    @property
    def secondary_mass(self) -> float:
        return self.mass_ratio * self.primary_mass

    @property
    def cloud_mass(self) -> float:
        return self.cloud_mass_fraction * self.primary_mass

    @property
    def alpha_secondary(self) -> float:
        return self.mass_ratio * self.alpha

    @property
    def boson_mass(self) -> float:
        return self.alpha / self.primary_mass

    @property
    def bohr_radius_primary(self) -> float:
        return self.primary_mass / self.alpha**2

    @property
    def bohr_radius_secondary(self) -> float:
        return self.bohr_radius_primary / self.mass_ratio

    @property
    def energy_scale(self) -> float:
        return self.boson_mass * self.alpha**2


@dataclass(frozen=True)
class ModelTables:
    """Integrated GM operators and cloud moments in the Python axis convention."""

    separation: FloatArray
    hamiltonian_dimless: FloatArray
    omega_lz_dimless: FloatArray
    dh_domega: FloatArray
    cloud_center: FloatArray
    cloud_quadrupole: FloatArray

    def __post_init__(self) -> None:
        n = self.separation.size
        expected = {
            "hamiltonian_dimless": (n, 6, 6),
            "omega_lz_dimless": (n, 6, 6),
            "dh_domega": (n, 6, 6),
            "cloud_center": (n, 6),
            "cloud_quadrupole": (n, 6, 3, 3),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")


@dataclass(frozen=True)
class DetectionGrid:
    """Precomputed mass/alpha scan used by the detection-region figure."""

    primary_masses: FloatArray
    alphas: FloatArray
    snr: FloatArray
    chirp_summary: FloatArray

    def __post_init__(self) -> None:
        expected = (self.primary_masses.size, self.alphas.size)
        if self.snr.shape != expected:
            raise ValueError(f"SNR grid has shape {self.snr.shape}, expected {expected}")
        if self.chirp_summary.shape[:2] != expected:
            raise ValueError(
                "chirp grid leading shape is "
                f"{self.chirp_summary.shape[:2]}, expected {expected}"
            )
        if self.chirp_summary.shape[-1] < 5:
            raise ValueError("chirp summaries must contain at least five columns")


def _number_label(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


class DataRepository:
    """Standalone NPZ cache with automatic pure-Python model-table generation."""

    FORMAT_VERSION: Final = 1

    def __init__(self, data_directory: str | Path) -> None:
        self.data_directory = Path(data_directory).expanduser().resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "DataRepository":
        return cls(Path(project_root) / "data")

    def model_cache_path(
        self,
        *,
        mass_ratio: float,
        number: int,
        start: float,
        end: float,
    ) -> Path:
        return self.data_directory / (
            f"gm_tables_v{self.FORMAT_VERSION}_q{_number_label(mass_ratio)}"
            f"_n{number}_x{_number_label(start)}_{_number_label(end)}.npz"
        )

    def _read_model_cache(self, path: Path) -> ModelTables:
        with np.load(path, allow_pickle=False) as cache:
            version = int(cache["format_version"])
            if version != self.FORMAT_VERSION:
                raise ValueError(
                    f"unsupported cache format {version} in {path}; "
                    f"expected {self.FORMAT_VERSION}"
                )
            return ModelTables(
                separation=np.asarray(cache["separation"], dtype=np.float64),
                hamiltonian_dimless=np.asarray(
                    cache["hamiltonian_dimless"], dtype=np.float64
                ),
                omega_lz_dimless=np.asarray(
                    cache["omega_lz_dimless"], dtype=np.float64
                ),
                dh_domega=np.asarray(cache["dh_domega"], dtype=np.float64),
                cloud_center=np.asarray(cache["cloud_center"], dtype=np.float64),
                cloud_quadrupole=np.asarray(
                    cache["cloud_quadrupole"], dtype=np.float64
                ),
            )

    def _write_model_cache(
        self,
        path: Path,
        mass_ratio: float,
        tables: ModelTables,
        settings: IntegrationSettings,
    ) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    format_version=np.asarray(self.FORMAT_VERSION),
                    mass_ratio=np.asarray(mass_ratio),
                    operator_order=np.asarray(settings.operator_order),
                    moment_order=np.asarray(settings.moment_order),
                    separation=tables.separation,
                    hamiltonian_dimless=tables.hamiltonian_dimless,
                    omega_lz_dimless=tables.omega_lz_dimless,
                    dh_domega=tables.dh_domega,
                    cloud_center=tables.cloud_center,
                    cloud_quadrupole=tables.cloud_quadrupole,
                )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_model_tables(
        self,
        *,
        mass_ratio: float = 0.99,
        number: int = 200,
        start: float = 10.0,
        end: float = 38.0,
        compute_if_missing: bool = True,
        integration_settings: IntegrationSettings | None = None,
    ) -> ModelTables:
        """Load a model cache, or integrate and atomically save it when absent."""

        if number < 1:
            raise ValueError("number must be positive")
        if end < start:
            raise ValueError("end must not be smaller than start")
        path = self.model_cache_path(
            mass_ratio=mass_ratio, number=number, start=start, end=end
        )
        if path.is_file():
            return self._read_model_cache(path)
        if not compute_if_missing:
            raise FileNotFoundError(f"model cache is missing: {path}")

        settings = integration_settings or IntegrationSettings()
        separations = np.linspace(start, end, number, dtype=np.float64)
        if settings.verbose:
            print(f"cache miss: computing {path.name} with pure-Python integration")
        tables = compute_model_tables(mass_ratio, separations, settings)
        self._write_model_cache(path, mass_ratio, tables, settings)
        return tables

    def load_detection_grid(
        self,
        *,
        observation_years: int = 10,
        cloud_mass_fraction: float = 0.01,
        primary_masses: ArrayLike | None = None,
        alphas: ArrayLike | None = None,
        mass_ratio: float = 0.99,
        redshift: float = 0.054,
        chirp_threshold: float = 0.04,
        compute_if_missing: bool = True,
        workers: int = 1,
        integration_settings: IntegrationSettings | None = None,
    ) -> DetectionGrid:
        """Load a population scan, or compute and save it when absent."""

        requested_masses = np.asarray(
            np.arange(10.0, 101.0) if primary_masses is None else primary_masses,
            dtype=np.float64,
        )
        requested_alphas = np.asarray(
            np.arange(0.01, 0.1600001, 0.001) if alphas is None else alphas,
            dtype=np.float64,
        )
        if requested_masses.ndim != 1 or requested_masses.size == 0:
            raise ValueError("primary_masses must be a non-empty 1D array")
        if requested_alphas.ndim != 1 or requested_alphas.size == 0:
            raise ValueError("alphas must be a non-empty 1D array")
        path = self.data_directory / (
            f"gm_detection_v{self.FORMAT_VERSION}_q{_number_label(mass_ratio)}"
            f"_z{_number_label(redshift)}_t{_number_label(observation_years)}"
            f"_mc{_number_label(cloud_mass_fraction)}"
            f"_d{_number_label(chirp_threshold)}"
            f"_m{_number_label(requested_masses[0])}"
            f"_{_number_label(requested_masses[-1])}_{requested_masses.size}"
            f"_a{_number_label(requested_alphas[0])}"
            f"_{_number_label(requested_alphas[-1])}_{requested_alphas.size}.npz"
        )
        if path.is_file():
            with np.load(path, allow_pickle=False) as cache:
                masses = np.asarray(cache["primary_masses"], dtype=np.float64)
                alpha_values = np.asarray(cache["alphas"], dtype=np.float64)
                snr = np.asarray(cache["snr"], dtype=np.float64)
                chirp = np.asarray(cache["chirp_summary"], dtype=np.float64)
            grid = DetectionGrid(masses, alpha_values, snr, chirp)
        elif not compute_if_missing:
            raise FileNotFoundError(
                f"detection-grid cache is missing: {path}"
            )
        else:
            grid = compute_detection_grid(
                self,
                mass_ratio=mass_ratio,
                primary_masses=requested_masses,
                alphas=requested_alphas,
                cloud_mass_fraction=cloud_mass_fraction,
                redshift=redshift,
                chirp_threshold=chirp_threshold,
                observation_years=observation_years,
                workers=workers,
                integration_settings=integration_settings,
            )
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        format_version=np.asarray(self.FORMAT_VERSION),
                        observation_years=np.asarray(observation_years),
                        cloud_mass_fraction=np.asarray(cloud_mass_fraction),
                        primary_masses=grid.primary_masses,
                        alphas=grid.alphas,
                        snr=grid.snr,
                        chirp_summary=grid.chirp_summary,
                    )
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        np.testing.assert_allclose(grid.primary_masses, requested_masses)
        np.testing.assert_allclose(grid.alphas, requested_alphas)
        return grid


@dataclass(frozen=True)
class EvolutionResult:
    """Sampled inspiral solution for one eigenstate."""

    time_years: FloatArray
    separation: FloatArray
    gw_frequency_planck: FloatArray
    chirp_mass: FloatArray
    chirp_mass_reference: float
    chirp_deviation: FloatArray
    state: int

    @property
    def gw_frequency_hz(self) -> FloatArray:
        return self.gw_frequency_planck / PLANCK_TIME_S


@dataclass(frozen=True)
class ChirpSummary:
    """Python equivalent of the ten values returned by ``GMMchirpSummary``."""

    time_at_max: float
    maximum_deviation: float
    left_time: float
    right_time: float
    width_years: float
    frequency_at_max_hz: float
    left_frequency_hz: float
    right_frequency_hz: float
    separation_at_max: float
    integrated_growth: float

    def as_array(self) -> FloatArray:
        return np.asarray(
            [
                self.time_at_max,
                self.maximum_deviation,
                self.left_time,
                self.right_time,
                self.width_years,
                self.frequency_at_max_hz,
                self.left_frequency_hz,
                self.right_frequency_hz,
                self.separation_at_max,
                self.integrated_growth,
            ],
            dtype=np.float64,
        )


class GMModel:
    """GM observables computed from cached or freshly integrated model tables."""

    def __init__(self, parameters: GMParameters, tables: ModelTables) -> None:
        self.parameters = parameters
        self.tables = tables

    @classmethod
    def from_repository(
        cls,
        repository: DataRepository,
        parameters: GMParameters | None = None,
        *,
        number: int = 200,
        start: float = 10.0,
        end: float = 38.0,
        compute_if_missing: bool = True,
        integration_settings: IntegrationSettings | None = None,
    ) -> "GMModel":
        params = parameters or GMParameters()
        tables = repository.load_model_tables(
            mass_ratio=params.mass_ratio,
            number=number,
            start=start,
            end=end,
            compute_if_missing=compute_if_missing,
            integration_settings=integration_settings,
        )
        return cls(params, tables)

    @property
    def separation(self) -> FloatArray:
        return self.tables.separation

    @cached_property
    def eigenvalues(self) -> FloatArray:
        values, _ = np.linalg.eigh(
            self.tables.hamiltonian_dimless * self.parameters.energy_scale
        )
        return values

    @cached_property
    def eigenvectors(self) -> ComplexArray:
        _, vectors = np.linalg.eigh(
            self.tables.hamiltonian_dimless * self.parameters.energy_scale
        )
        # np.linalg.eigh stores eigenvectors in columns; Mathematica and the
        # formulas below use one eigenvector per row.
        return np.swapaxes(vectors, 1, 2).astype(np.complex128)

    @cached_property
    def eigenvalue_derivatives(self) -> FloatArray:
        return numerical_derivative(self.eigenvalues, self.separation)

    @staticmethod
    def _expectation(vectors: ComplexArray, matrices: ArrayLike) -> FloatArray:
        matrix_array = np.asarray(matrices)
        result = np.einsum(
            "nki,nij,nkj->nk", vectors.conj(), matrix_array, vectors, optimize=True
        )
        return np.real_if_close(result, tol=1_000).real.astype(np.float64)

    @cached_property
    def dh_domega_expectation(self) -> FloatArray:
        return self._expectation(self.eigenvectors, self.tables.dh_domega)

    @cached_property
    def gw_frequency_squared(self) -> FloatArray:
        """GW frequency squared, with shape ``(n_separations, 6)``."""

        p = self.parameters
        x = self.separation[:, None]
        dedr = self.eigenvalue_derivatives + (
            1.5
            * p.energy_scale
            * np.sqrt(1.0 + p.mass_ratio)
            * x ** (-1.5)
            * self.dh_domega_expectation
            / x
        )
        x_system = p.cloud_mass * self.tables.cloud_center / (
            p.cloud_mass + p.primary_mass + p.secondary_mass
        )
        d1 = p.mass_ratio / (1.0 + p.mass_ratio) * x + x_system
        d2 = 1.0 / (1.0 + p.mass_ratio) * x - x_system
        d1_derivative = numerical_derivative(d1, self.separation)
        d2_derivative = numerical_derivative(d2, self.separation)

        numerator = (
            p.primary_mass * p.secondary_mass / (x**2 * p.bohr_radius_primary)
            + dedr * (p.cloud_mass / p.boson_mass)
        )
        denominator = (
            p.primary_mass * d1 * d1_derivative
            + p.secondary_mass * d2 * d2_derivative
        ) * p.bohr_radius_primary**2
        result = numerator / denominator / np.pi**2
        if np.any(result <= 0.0):
            raise FloatingPointError("computed non-positive GW frequency squared")
        return result

    @cached_property
    def gw_frequency(self) -> FloatArray:
        return np.sqrt(self.gw_frequency_squared)

    @cached_property
    def mass_quadrupole(self) -> FloatArray:
        p = self.parameters
        x = self.separation[:, None]
        x_cloud = self.tables.cloud_center
        q_bbh = (
            p.primary_mass
            * (
                p.mass_ratio / (1.0 + p.mass_ratio) * x * p.bohr_radius_primary
                + p.cloud_mass
                * x_cloud
                * p.bohr_radius_primary
                / (p.primary_mass + p.secondary_mass + p.cloud_mass)
            )
            ** 2
            + p.secondary_mass
            * (
                1.0 / (1.0 + p.mass_ratio) * x * p.bohr_radius_primary
                - p.cloud_mass
                * x_cloud
                * p.bohr_radius_primary
                / (p.primary_mass + p.secondary_mass + p.cloud_mass)
            )
            ** 2
        )
        result = (
            p.cloud_mass
            * self.tables.cloud_quadrupole
            * p.bohr_radius_primary**2
        ).copy()
        result[:, :, 0, 0] += q_bbh
        return result

    @cached_property
    def gr_power(self) -> FloatArray:
        """Quadrupole GW power from ``GetGRPower``."""

        q = self.mass_quadrupole
        bracket = (
            16.0 * q[:, :, 0, 0] ** 2
            + 72.0 * q[:, :, 0, 1] ** 2
            + q[:, :, 0, 2] ** 2
            + 8.0 * q[:, :, 1, 1] ** 2
            - 16.0 * q[:, :, 0, 0] * (q[:, :, 0, 1] + q[:, :, 1, 1])
            + q[:, :, 1, 2] ** 2
        )
        return (2.0 / 5.0) * bracket * (np.pi**2 * self.gw_frequency_squared) ** 3

    @cached_property
    def cloud_energies_lab(self) -> FloatArray:
        correction = self._expectation(
            self.eigenvectors, self.tables.omega_lz_dimless
        )
        return self.eigenvalues + correction * self.parameters.energy_scale

    @cached_property
    def cloud_energy_derivatives_lab(self) -> FloatArray:
        return numerical_derivative(self.cloud_energies_lab, self.separation)

    @cached_property
    def system_energies_lab(self) -> FloatArray:
        """Total cloud+binary energy used by the inspiral ODE."""

        p = self.parameters
        x = self.separation[:, None]
        rho1 = p.mass_ratio / (1.0 + p.mass_ratio) * x
        rho2 = 1.0 / (1.0 + p.mass_ratio) * x
        x_system = (
            p.cloud_mass * self.tables.cloud_center
            - p.primary_mass * rho1
            + p.secondary_mass * rho2
        ) / (p.cloud_mass + p.primary_mass + p.secondary_mass)
        d1 = p.mass_ratio / (1.0 + p.mass_ratio) * x + x_system
        d2 = 1.0 / (1.0 + p.mass_ratio) * x - x_system
        return (
            (p.cloud_mass / p.boson_mass) * self.cloud_energies_lab
            - p.primary_mass
            * p.secondary_mass
            / (x * p.bohr_radius_primary)
            + 0.5
            * p.primary_mass
            * (np.pi**2 * self.gw_frequency_squared)
            * (d1 * p.bohr_radius_primary) ** 2
            + 0.5
            * p.secondary_mass
            * (np.pi**2 * self.gw_frequency_squared)
            * (d2 * p.bohr_radius_primary) ** 2
        )

    def chirp_mass_reference(self, state: int = 0) -> float:
        self._validate_state(state)
        p = self.parameters
        if state < 3:
            masses = ((p.primary_mass + p.cloud_mass), p.secondary_mass)
        else:
            masses = (p.primary_mass, (p.secondary_mass + p.cloud_mass))
        return (masses[0] * masses[1]) ** (3.0 / 5.0) / (
            p.primary_mass + p.secondary_mass + p.cloud_mass
        ) ** (1.0 / 5.0)

    def evolve(
        self,
        state: int = 0,
        *,
        samples: int = 2_000,
        rtol: float = 2e-10,
        atol: float = 1e-9,
    ) -> EvolutionResult:
        """Integrate the separation ODE, returning an evenly sampled solution."""

        self._validate_state(state)
        if samples < 20:
            raise ValueError("samples must be at least 20")
        x = self.separation
        energy = CubicSpline(x, self.system_energies_lab[:, state])
        energy_derivative = energy.derivative()
        power = CubicSpline(x, self.gr_power[:, state])
        frequency = CubicSpline(x, self.gw_frequency[:, state])
        frequency_derivative = frequency.derivative()
        initial = float(x[-3])  # Mathematica: xValues[[Num - 2]]
        requested_stop = float(x[1])

        # The adiabatic equation dR/dt = -P/(dE/dR) is single-valued only on
        # a branch where dE/dR stays positive.  Unequal-mass configurations
        # can develop an internal energy turning point before reaching the
        # nominal lower boundary (q=0.5, state=0 is one example).  End at the
        # first tabulated radius outside that turning point rather than asking
        # solve_ivp to cross the resulting pole.
        if float(energy_derivative(initial)) <= 0.0:
            raise FloatingPointError(
                "the initial radius is outside a stable inspiral branch"
            )
        derivative_roots = np.asarray(
            energy_derivative.roots(extrapolate=False), dtype=np.float64
        )
        internal_roots = derivative_roots[
            np.isfinite(derivative_roots)
            & (derivative_roots > requested_stop)
            & (derivative_roots < initial)
        ]
        stop = requested_stop
        if internal_roots.size:
            turning_point = float(np.max(internal_roots))
            stop_index = int(np.searchsorted(x, turning_point, side="right"))
            if stop_index >= x.size - 2:
                raise FloatingPointError(
                    "no resolved stable inspiral interval below the initial radius"
                )
            stop = float(x[stop_index])

        def radial_velocity(radius: float | NDArray[np.float64]) -> FloatArray:
            return np.asarray(
                -power(radius) * PLANCK_TIMES_PER_YEAR / energy_derivative(radius),
                dtype=np.float64,
            )

        def time_density(radius: float | NDArray[np.float64]) -> FloatArray:
            return np.asarray(
                energy_derivative(radius)
                / (power(radius) * PLANCK_TIMES_PER_YEAR),
                dtype=np.float64,
            )

        # This value only sets the solve_ivp time span and maximum step; the
        # event supplies the final time.  Composite Simpson integration is
        # stable across the many spline segments and avoids a misleading
        # QUADPACK roundoff warning at unnecessarily tight tolerances.
        probe = np.linspace(stop, initial, 4_097)
        density_values = time_density(probe)
        if not np.all(np.isfinite(density_values)):
            raise FloatingPointError("inspiral-time density became non-finite")
        estimate = float(simpson(density_values, x=probe))
        if not np.isfinite(estimate) or estimate <= 0.0:
            raise FloatingPointError(f"invalid inspiral-time estimate: {estimate}")

        def rhs(_time: float, radius: FloatArray) -> FloatArray:
            return radial_velocity(radius)

        def stop_event(_time: float, radius: FloatArray) -> float:
            return float(radius[0] - stop)

        stop_event.terminal = True  # type: ignore[attr-defined]
        stop_event.direction = -1.0  # type: ignore[attr-defined]
        solution = solve_ivp(
            rhs,
            (0.0, 1.25 * estimate),
            np.asarray([initial]),
            events=stop_event,
            dense_output=True,
            rtol=rtol,
            atol=atol,
            max_step=estimate / 1_500.0,
        )
        if not solution.success or solution.t_events[0].size != 1:
            raise RuntimeError(f"inspiral integration did not reach its event: {solution.message}")

        final_time = float(solution.t_events[0][0])
        time = np.linspace(0.0, final_time, samples)
        radius = np.asarray(solution.sol(time)[0], dtype=np.float64)
        freq = np.asarray(frequency(radius), dtype=np.float64)
        drdt = radial_velocity(radius)
        chirp_base = (
            (5.0 / 96.0)
            * frequency_derivative(radius)
            * (drdt / PLANCK_TIMES_PER_YEAR)
            / (np.pi ** (8.0 / 3.0) * freq ** (11.0 / 3.0))
        )
        if np.any(chirp_base <= 0.0):
            raise FloatingPointError("chirp-mass base became non-positive")
        chirp_mass = chirp_base ** (3.0 / 5.0)
        reference = self.chirp_mass_reference(state)
        return EvolutionResult(
            time_years=time,
            separation=radius,
            gw_frequency_planck=freq,
            chirp_mass=chirp_mass,
            chirp_mass_reference=reference,
            chirp_deviation=chirp_mass / reference - 1.0,
            state=state,
        )

    def chirp_summary(
        self,
        threshold: float,
        state: int = 0,
        *,
        samples: int = 4_000,
        spin_primary: float = 1.0,
        spin_secondary: float = 1.0,
    ) -> ChirpSummary:
        """Return the summary and integrated growth used by the main notebook."""

        evolution = self.evolve(state, samples=samples)
        t = evolution.time_years
        deviation_spline = CubicSpline(t, evolution.chirp_deviation)
        radius_spline = CubicSpline(t, evolution.separation)
        frequency_spline = CubicSpline(t, evolution.gw_frequency_planck)
        peak_seed = int(np.argmax(evolution.chirp_deviation))
        left_bound = t[max(0, peak_seed - 2)]
        right_bound = t[min(t.size - 1, peak_seed + 2)]
        optimum = minimize_scalar(
            lambda value: -float(deviation_spline(value)),
            bounds=(left_bound, right_bound),
            method="bounded",
            options={"xatol": max(1e-6, t[-1] * 1e-13)},
        )
        time_at_max = float(optimum.x)
        maximum = float(deviation_spline(time_at_max))
        target = maximum - threshold

        left_time = 0.0
        if float(deviation_spline(0.0)) <= target:
            left_time = float(
                brentq(
                    lambda value: float(deviation_spline(value) - target),
                    0.0,
                    time_at_max,
                )
            )
        right_time = float(t[-1])
        if float(deviation_spline(t[-1])) <= target:
            right_time = float(
                brentq(
                    lambda value: float(deviation_spline(value) - target),
                    time_at_max,
                    float(t[-1]),
                )
            )

        growth_values = self.eigensystem_with_growth(
            spin_primary=spin_primary, spin_secondary=spin_secondary
        )[0][:, state].imag
        growth_spline = CubicSpline(self.separation, growth_values)
        growth = quad(
            lambda value: float(growth_spline(radius_spline(value)))
            * PLANCK_TIMES_PER_YEAR,
            0.0,
            time_at_max,
            epsrel=1e-7,
            limit=300,
        )[0]

        frequency_at_max = float(frequency_spline(time_at_max) / PLANCK_TIME_S)
        left_frequency = float(frequency_spline(left_time) / PLANCK_TIME_S)
        right_frequency = float(frequency_spline(right_time) / PLANCK_TIME_S)
        return ChirpSummary(
            time_at_max=time_at_max,
            maximum_deviation=maximum,
            left_time=left_time,
            right_time=right_time,
            width_years=right_time - left_time,
            frequency_at_max_hz=frequency_at_max,
            left_frequency_hz=left_frequency,
            right_frequency_hz=right_frequency,
            separation_at_max=float(radius_spline(time_at_max)),
            integrated_growth=float(growth),
        )

    def growth_width(
        self,
        state: Sequence[int],
        spin: float,
        black_hole_mass: float,
        alpha: float,
    ) -> float:
        """Growth width used in ``GMMain_v2.0.nb`` (including its correction)."""

        n, ell, m = (int(value) for value in state)
        if not (0.0 <= spin <= 1.0):
            raise ValueError("dimensionless spin must lie in [0, 1]")
        r_plus = 1.0 + np.sqrt(1.0 - spin**2)
        omega_h = spin / (2.0 * black_hole_mass * r_plus)
        level_frequency = self.parameters.boson_mass * (
            1.0 - alpha**2 / (2.0 * n**2)
        )
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

    def eigensystem_with_growth(
        self,
        *,
        spin_primary: float = 1.0,
        spin_secondary: float = 1.0,
    ) -> tuple[ComplexArray, ComplexArray]:
        """Eigensystem of the non-Hermitian Hamiltonian with growth widths."""

        p = self.parameters
        states = ((2, 0, 0), (2, 1, -1), (2, 1, 1))
        widths = np.asarray(
            [
                *(
                    self.growth_width(state, spin_primary, p.primary_mass, p.alpha)
                    for state in states
                ),
                *(
                    self.growth_width(
                        state,
                        spin_secondary,
                        p.secondary_mass,
                        p.alpha_secondary,
                    )
                    for state in states
                ),
            ]
        )
        hamiltonian = (
            self.tables.hamiltonian_dimless.astype(np.complex128) * p.energy_scale
        )
        hamiltonian = hamiltonian.copy()
        diagonal = np.arange(6)
        hamiltonian[:, diagonal, diagonal] += 1j * widths

        all_values: list[ComplexArray] = []
        all_vectors: list[ComplexArray] = []
        for matrix in hamiltonian:
            values, vectors = np.linalg.eig(matrix)
            order = np.argsort(-np.abs(values))
            all_values.append(values[order])
            all_vectors.append(vectors[:, order].T)
        return np.asarray(all_values), np.asarray(all_vectors)

    def validation_metrics(self) -> dict[str, float | tuple[int, ...]]:
        """Compact parser/numerics diagnostics suitable for tests and notebooks."""

        h = self.tables.hamiltonian_dimless
        omega_lz = self.tables.omega_lz_dimless
        return {
            "separation_shape": self.separation.shape,
            "hamiltonian_shape": h.shape,
            "cloud_center_shape": self.tables.cloud_center.shape,
            "cloud_quadrupole_shape": self.tables.cloud_quadrupole.shape,
            "hamiltonian_hermiticity_max_abs": float(
                np.max(np.abs(h - np.swapaxes(h, 1, 2)))
            ),
            "omega_lz_hermiticity_max_abs": float(
                np.max(np.abs(omega_lz - np.swapaxes(omega_lz, 1, 2)))
            ),
            "eigen_residual_max_abs": self._eigen_residual(),
            "gw_frequency_squared_min": float(np.min(self.gw_frequency_squared)),
            "gr_power_min": float(np.min(self.gr_power)),
        }

    def _eigen_residual(self) -> float:
        h = self.tables.hamiltonian_dimless * self.parameters.energy_scale
        residual = np.einsum("nij,nkj->nki", h, self.eigenvectors) - (
            self.eigenvalues[:, :, None] * self.eigenvectors
        )
        scale = np.max(np.abs(h))
        return float(np.max(np.abs(residual)) / scale)

    @staticmethod
    def _validate_state(state: int) -> None:
        if state not in range(6):
            raise ValueError("state must be a zero-based index in range(6)")


def sn_acceleration(frequency_hz: ArrayLike, model: str = "N2") -> FloatArray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    levels = {"N1": 9e-28, "N2": 9e-30}
    if model not in levels:
        raise ValueError(f"unknown acceleration-noise model: {model}")
    return levels[model] * (2.0 * np.pi * frequency) ** -4 * (
        1.0 + 1.0e-4 / frequency
    )


def sn_shot(_frequency_hz: ArrayLike, arm_model: str = "A5") -> FloatArray:
    levels = {"A1": 1.98e-23, "A2": 2.22e-23, "A5": 2.96e-23}
    if arm_model not in levels:
        raise ValueError(f"unknown arm model: {arm_model}")
    return np.asarray(levels[arm_model], dtype=np.float64)


def sn_other(_frequency_hz: ArrayLike) -> FloatArray:
    return np.asarray(2.65e-23, dtype=np.float64)


def sn_lisa(
    frequency_hz: ArrayLike,
    acceleration_model: str = "N2",
    arm_model: str = "A5",
) -> FloatArray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    arm_lengths = {"A1": 1e9, "A2": 2e9, "A5": 5e9}
    if arm_model not in arm_lengths:
        raise ValueError(f"unknown arm model: {arm_model}")
    arm = arm_lengths[arm_model]
    return (
        (20.0 / 3.0)
        * (
            4.0 * sn_acceleration(frequency, acceleration_model)
            + sn_shot(frequency, arm_model)
            + sn_other(frequency)
        )
        / arm**2
        * (1.0 + (frequency / (0.41 * SPEED_OF_LIGHT / (2.0 * arm))) ** 2)
    )


def sn_galactic(frequency_hz: ArrayLike, arm_model: str = "A5") -> FloatArray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    result = np.zeros_like(frequency)
    if arm_model == "A1":
        pieces = (
            (1e-5, 5.3e-4, 1.55206e-43, -2.1),
            (5.3e-4, 2.2e-3, 2.9714e-47, -3.235),
            (2.2e-3, 4e-3, 1.517e-51, -4.85),
            (4e-3, 5.3e-3, 6.706e-58, -7.5),
            (5.3e-3, 1e-2, 2.39835e-86, -20.0),
        )
    elif arm_model == "A5":
        pieces = (
            (1e-5, 1e-3, 10.0**-44.62, -2.3),
            (1e-3, 10.0**-2.7, 10.0**-50.92, -4.4),
            (10.0**-2.7, 10.0**-2.4, 10.0**-62.8, -8.8),
            (10.0**-2.4, 1e-2, 10.0**-89.68, -20.0),
        )
    else:
        raise ValueError("galactic noise is defined only for A1 and A5")
    for lower, upper, coefficient, exponent in pieces:
        mask = (frequency >= lower) & (frequency < upper)
        result[mask] = coefficient * frequency[mask] ** exponent
    return result


def sn_lisa_default(frequency_hz: ArrayLike) -> FloatArray:
    return sn_lisa(frequency_hz, "N2", "A5") + sn_galactic(frequency_hz, "A5")


def luminosity_distance_planck(redshift: float) -> float:
    """Luminosity distance in Planck lengths, matching ``DL`` in the notebook."""

    integral = quad(
        lambda x: 1.0 / np.sqrt(OMEGA_M * (1.0 + x) ** 3 + OMEGA_LAMBDA),
        0.0,
        redshift,
        epsrel=1e-12,
    )[0]
    return (
        (MPC_M / PLANCK_LENGTH_M)
        * (1.0 + redshift)
        * SPEED_OF_LIGHT
        / HUBBLE_KM_S_MPC
        * integral
    )


def snr_bbh_range(
    mass_ratio: float,
    primary_mass_solar: float,
    redshift: float,
    luminosity_distance_m: float,
    minimum_radius_m: float,
    observation_years: float,
    noise: Callable[[ArrayLike], FloatArray] = sn_lisa_default,
) -> float:
    """BBH signal-to-noise ratio translated from ``SNRBBHRange``."""

    total_mass = (1.0 + mass_ratio) * primary_mass_solar * SOLAR_MASS_KG
    chirp_mass = (
        mass_ratio ** (3.0 / 5.0)
        / (1.0 + mass_ratio) ** (1.0 / 5.0)
        * primary_mass_solar
        * SOLAR_MASS_KG
    )
    max_frequency = (
        np.sqrt(NEWTON_G * total_mass / minimum_radius_m**3)
        / (np.pi * (1.0 + redshift))
    )
    coalescence_time = (
        5.0
        * SPEED_OF_LIGHT**5
        / (
            256.0
            * ((1.0 + redshift) * max_frequency) ** (8.0 / 3.0)
            * (NEWTON_G * chirp_mass) ** (5.0 / 3.0)
            * np.pi ** (8.0 / 3.0)
        )
    )
    initial_time = coalescence_time + observation_years * SECONDS_PER_YEAR
    min_frequency = (
        (
            5.0
            * SPEED_OF_LIGHT**5
            / (
                256.0
                * initial_time
                * (NEWTON_G * chirp_mass) ** (5.0 / 3.0)
                * np.pi ** (8.0 / 3.0)
            )
        )
        ** (3.0 / 8.0)
        / (1.0 + redshift)
    )

    def integrand(frequency: float) -> float:
        characteristic_squared = (
            2.0
            / (np.pi**2 * luminosity_distance_m**2)
            * np.pi ** (2.0 / 3.0)
            / (3.0 * SPEED_OF_LIGHT**3)
            * ((1.0 + redshift) * NEWTON_G * chirp_mass) ** (5.0 / 3.0)
            * frequency ** (-1.0 / 3.0)
        )
        return characteristic_squared / (frequency**2 * float(noise(frequency)))

    return float(np.sqrt(quad(integrand, min_frequency, max_frequency, limit=300)[0]))


_DETECTION_TABLES: ModelTables | None = None
_DETECTION_COMMON: tuple[float, float, float, float, float, float] | None = None


def _initialize_detection_worker(
    tables: ModelTables,
    common: tuple[float, float, float, float, float, float],
) -> None:
    global _DETECTION_TABLES, _DETECTION_COMMON
    _DETECTION_TABLES = tables
    _DETECTION_COMMON = common


def _compute_detection_point(
    point: tuple[float, float],
) -> tuple[FloatArray, float]:
    if _DETECTION_TABLES is None or _DETECTION_COMMON is None:
        raise RuntimeError("detection worker was not initialized")
    mass, alpha = point
    q, cloud_fraction, redshift, distance_m, threshold, observation_years = (
        _DETECTION_COMMON
    )
    parameters = GMParameters(
        mass_ratio=q,
        primary_mass_solar=mass,
        alpha=alpha,
        cloud_mass_fraction=cloud_fraction,
    )
    model = GMModel(parameters, _DETECTION_TABLES)
    summary = model.chirp_summary(threshold, state=0, samples=4_000)
    bohr_radius_m = (
        NEWTON_G
        * mass
        * SOLAR_MASS_KG
        / (SPEED_OF_LIGHT**2 * alpha**2)
    )
    snr = snr_bbh_range(
        q,
        mass,
        redshift,
        distance_m,
        summary.separation_at_max * bohr_radius_m,
        observation_years,
    )
    return summary.as_array(), snr


def compute_detection_grid(
    repository: DataRepository,
    *,
    mass_ratio: float = 0.99,
    primary_masses: ArrayLike | None = None,
    alphas: ArrayLike | None = None,
    cloud_mass_fraction: float = 0.01,
    redshift: float = 0.054,
    chirp_threshold: float = 0.04,
    observation_years: float = 10.0,
    workers: int = 1,
    integration_settings: IntegrationSettings | None = None,
    verbose: bool = True,
) -> DetectionGrid:
    """Compute the mass/alpha chirp and SNR scan used by the contour plot."""

    masses = np.asarray(
        np.arange(10.0, 101.0) if primary_masses is None else primary_masses,
        dtype=np.float64,
    )
    alpha_values = np.asarray(
        np.arange(0.01, 0.1600001, 0.001) if alphas is None else alphas,
        dtype=np.float64,
    )
    if masses.ndim != 1 or alpha_values.ndim != 1:
        raise ValueError("primary_masses and alphas must be one-dimensional")
    tables = repository.load_model_tables(
        mass_ratio=mass_ratio,
        integration_settings=integration_settings,
    )
    distance_m = luminosity_distance_planck(redshift) * PLANCK_LENGTH_M
    common = (
        float(mass_ratio),
        float(cloud_mass_fraction),
        float(redshift),
        float(distance_m),
        float(chirp_threshold),
        float(observation_years),
    )
    points = [
        (float(mass), float(alpha))
        for mass in masses
        for alpha in alpha_values
    ]
    resolved_workers = (
        min(64, max(1, os.cpu_count() or 1)) if workers == -1 else workers
    )
    if resolved_workers < 1:
        raise ValueError("workers must be -1 or a positive integer")

    if resolved_workers == 1:
        _initialize_detection_worker(tables, common)
        iterator = map(_compute_detection_point, points)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=min(resolved_workers, len(points)),
            initializer=_initialize_detection_worker,
            initargs=(tables, common),
        )
        iterator = pool.map(_compute_detection_point, points, chunksize=1)
    results = []
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if verbose and index % alpha_values.size == 0:
                print(f"computed detection scan row {index // alpha_values.size}/{masses.size}")
    finally:
        if pool is not None:
            pool.shutdown()
    summaries, snr_values = zip(*results)
    return DetectionGrid(
        primary_masses=masses,
        alphas=alpha_values,
        snr=np.asarray(snr_values, dtype=np.float64).reshape(
            masses.size, alpha_values.size
        ),
        chirp_summary=np.asarray(summaries, dtype=np.float64).reshape(
            masses.size, alpha_values.size, -1
        ),
    )


def plot_detection_regions(
    grid: DetectionGrid,
    *,
    chirp_threshold: float = 0.04,
    detection_time_years: float = 10.0,
    snr_threshold: float = 13.0,
    ax=None,
):
    """Reproduce the red/green/blue detection overlay from input cell 11."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    x, y = np.meshgrid(grid.primary_masses, grid.alphas)
    conditions = (
        (grid.snr.T >= snr_threshold, "red", f"SNR ≥ {snr_threshold:g}"),
        (
            grid.chirp_summary[:, :, 1].T >= chirp_threshold,
            "green",
            rf"max $\Delta\mathcal{{M}}/\mathcal{{M}}_0$ ≥ {chirp_threshold:g}",
        ),
        (
            grid.chirp_summary[:, :, 4].T <= detection_time_years,
            "blue",
            rf"duration ≤ {detection_time_years:g} yr",
        ),
    )
    for condition, color, label in conditions:
        ax.contourf(
            x,
            y,
            condition.astype(float),
            levels=[0.5, 1.5],
            colors=[to_rgba(color, 0.3)],
        )
        ax.contour(x, y, condition.astype(float), levels=[0.5], colors=[color])
        ax.plot([], [], color=color, linewidth=2, label=label)
    ax.axhline(0.15, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel=r"$m_1/M_\odot$", ylabel=r"$\alpha$", xlim=(x.min(), x.max()))
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_evolution(
    evolution: EvolutionResult,
    *,
    threshold: float | None = None,
    ax=None,
):
    """Plot separation and chirp deviation in the style of input cell 14."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    else:
        axes = ax
    if len(axes) != 2:
        raise ValueError("ax must contain two matplotlib axes")
    label = rf"$\hat{{\eta}}_{evolution.state + 1}$"
    axes[0].plot(evolution.time_years, evolution.separation, label=label)
    axes[0].set(xlabel=r"$t/\mathrm{yr}$", ylabel=r"$R/r_{b,1}$")
    axes[0].legend(frameon=False)
    axes[1].plot(evolution.time_years, evolution.chirp_deviation, label=label)
    if threshold is not None:
        level = float(np.max(evolution.chirp_deviation) - threshold)
        above = np.flatnonzero(evolution.chirp_deviation >= level)
        axes[1].hlines(
            level,
            evolution.time_years[above[0]],
            evolution.time_years[above[-1]],
            color="tab:orange",
            linewidth=1.5,
        )
    axes[1].set(
        xlabel=r"$t/\mathrm{yr}$",
        ylabel=r"$\mathcal{M}/\mathcal{M}_0-1$",
    )
    axes[1].legend(frameon=False)
    return axes


def plot_growth_eigensystem(
    model: GMModel,
    *,
    spin_primary: float = 1.0,
    spin_secondary: float = 1.0,
    ax=None,
):
    """Plot real energies and absolute growth widths from input cell 17."""

    import matplotlib.pyplot as plt

    values, _ = model.eigensystem_with_growth(
        spin_primary=spin_primary, spin_secondary=spin_secondary
    )
    values = values / model.parameters.energy_scale
    if ax is None:
        _, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    else:
        axes = ax
    if len(axes) != 2:
        raise ValueError("ax must contain two matplotlib axes")
    for state in range(6):
        label = rf"$\hat{{\eta}}_{state + 1}$"
        axes[0].plot(model.separation, values[:, state].real, label=label)
        axes[1].plot(model.separation, np.abs(values[:, state].imag), label=label)
    axes[0].set(
        xlabel=r"$R/r_{b,1}$",
        ylabel=r"$\mathrm{Re}\,E_{\hat{\eta}_i}/(\mu\alpha_1^2)$",
    )
    axes[1].set(
        xlabel=r"$R/r_{b,1}$",
        ylabel=r"$|\mathrm{Im}\,E_{\hat{\eta}_i}|/(\mu\alpha_1^2)$",
        yscale="log",
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    return axes


__all__ = [
    "ChirpSummary",
    "DataRepository",
    "DetectionGrid",
    "EvolutionResult",
    "GMModel",
    "GMParameters",
    "IntegrationSettings",
    "ModelTables",
    "NEWTON_G",
    "PLANCK_LENGTH_M",
    "PLANCK_MASS_KG",
    "PLANCK_TIME_S",
    "PLANCK_TIMES_PER_YEAR",
    "SOLAR_MASS_PLANCK",
    "compute_detection_grid",
    "compute_model_tables",
    "hydrogenic_basis_states",
    "luminosity_distance_planck",
    "numerical_derivative",
    "plot_detection_regions",
    "plot_evolution",
    "plot_growth_eigensystem",
    "sn_acceleration",
    "sn_galactic",
    "sn_lisa",
    "sn_lisa_default",
    "sn_other",
    "sn_shot",
    "snr_bbh_range",
]
