"""Analytic dipole phase of each quantum path.

φ_j(I) = a_j·I + b_j/I·(qω − Ip/ℏ)², trajectories '1'..'6' (odd short, even long, returns 1 to 3), after Guo et al.,
J. Phys. B 51, 034006 (2018) and Wikmark et al., PNAS 116, 4779 (2019), who write α and γ for a and b.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import (
    c as C_0,
    e as E_CHARGE,
    epsilon_0 as EPS_0,
    fine_structure as ALPHA_FS,
    hbar as HBAR,
    m_e as M_E,
    physical_constants as _pc,
)

from .sfa import IP_AU

AU_ENERGY = _pc["atomic unit of energy"][0]

# trajectory: (c_a, c_b, c_up); Ip + c_up·Up is the classical cutoff of its return (1st, 2nd, 3rd)
_TRAJECTORIES = {
    "1": (0.0, 1.4 / 3.17, 3.17),
    "2": (1.0, -1.2 / 3.17, 3.17),
    "3": (1.0, 1.0 / 1.54, 1.54),
    "4": (1.4, -1.0 / 1.54, 1.54),
    "5": (1.4, 1.10 / 2.40, 2.40),
    "6": (2.0, -1.10 / 2.40, 2.40),
}


def _terms(q: int, trajectory: str, gas: str, wavelength: float) -> tuple[float, float, float]:
    """a_j [rad m²/W], b_j·(qω − Ip/ℏ)² [rad W/m²] and the cutoff intensity [W/m²], NaN below threshold."""
    c_a, c_b, c_up = _TRAJECTORIES[trajectory]
    omega0 = 2 * np.pi * C_0 / wavelength
    up_per_intensity = E_CHARGE**2 / (2 * EPS_0 * C_0 * M_E * omega0**2)  # Up [J] per W/m²
    excess = q * HBAR * omega0 - IP_AU[gas] * AU_ENERGY  # qħω − Ip [J]
    a = -c_a / (2 * np.pi) * ALPHA_FS * wavelength**3 / (M_E * C_0**3)
    b = c_b / 2 * C_0 * M_E / (ALPHA_FS * wavelength) * (excess / HBAR) ** 2
    if excess <= 0:
        warnings.warn(
            f"H{q} is below the ionization potential of {gas} at {wavelength * 1e9:.0f} nm, the analytic phase is NaN",
            stacklevel=3,
        )
        return a, b, np.nan
    return a, b, excess / (c_up * up_per_intensity)


def cutoff_intensity(q: int, trajectory: str, gas: str, wavelength: float) -> float:
    """Cutoff intensity of a trajectory, where qħω = Ip + c_up·Up.

    Parameters
    ----------
    q : int
        Harmonic order.
    trajectory : {"1", "2", "3", "4", "5", "6"}
        Quantum path, odd short and even long, returns 1 to 3.
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].

    Returns
    -------
    float
        Cutoff intensity [W/m²], NaN below threshold.

    Warns
    -----
    UserWarning
        If the harmonic is below threshold (qħω ≤ Ip).
    """
    return _terms(q, trajectory, gas, wavelength)[2]


def dipole_phase(I: ArrayLike, q: int, trajectory: str, gas: str, wavelength: float) -> np.ndarray:
    """Dipole phase φ_j(I) = a_j·I + b_j/I·(qω − Ip/ℏ)² of a trajectory.

    Parameters
    ----------
    I : array_like
        Intensity [W/m²].
    q : int
        Harmonic order.
    trajectory : {"1", "2", "3", "4", "5", "6"}
        Quantum path, odd short and even long, returns 1 to 3.
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].

    Returns
    -------
    ndarray
        Phase [rad], NaN below the cutoff intensity.

    Warns
    -----
    UserWarning
        If the harmonic is below threshold (qħω ≤ Ip).
    """
    a, b, I_cut = _terms(q, trajectory, gas, wavelength)
    I = np.where(np.asarray(I, dtype=float) >= I_cut, I, np.nan)
    return a * I + b / I


def d_dipole_phase_dI(I: ArrayLike, q: int, trajectory: str, gas: str, wavelength: float) -> np.ndarray:
    """Local slope α_j = ∂φ_j/∂I = a_j − b_j/I²·(qω − Ip/ℏ)² of the dipole phase of a trajectory.

    Parameters
    ----------
    I : array_like
        Intensity [W/m²].
    q : int
        Harmonic order.
    trajectory : {"1", "2", "3", "4", "5", "6"}
        Quantum path, odd short and even long, returns 1 to 3.
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].

    Returns
    -------
    ndarray
        α_j [rad m²/W], NaN below the cutoff intensity.

    Warns
    -----
    UserWarning
        If the harmonic is below threshold (qħω ≤ Ip).
    """
    a, b, I_cut = _terms(q, trajectory, gas, wavelength)
    I = np.where(np.asarray(I, dtype=float) >= I_cut, I, np.nan)
    return a - b / I**2


def harmonic_dipole(
    I: ArrayLike,
    q: int,
    trajectories: tuple[str, ...],
    gas: str,
    wavelength: float,
    weights: dict[str, float] | None = None,
    q_eff: float = 4,
) -> np.ndarray:
    """Harmonic dipole d_q(I) = I^(q_eff/2)·Σ_j w_j·exp(iφ_j(I)) of the chosen trajectories.

    Each phase is frozen at its cutoff value below its cutoff intensity.

    Parameters
    ----------
    I : array_like
        Intensity [W/m²].
    q : int
        Harmonic order.
    trajectories : tuple of str
        Quantum paths summed coherently, e.g. ("1", "2") for the short and long paths of the first return.
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].
    weights : dict of str to float, optional
        Amplitude w_j of each trajectory, 1 for those not given.
    q_eff : float, default 4
        Intensity-scaling exponent of the harmonic yield, ``|d|² ∝ I^q_eff``.

    Returns
    -------
    ndarray
        Complex d_q at each intensity.

    Warns
    -----
    UserWarning
        If the harmonic is below threshold (qħω ≤ Ip).
    """
    I, weights = np.asarray(I, dtype=float), weights or {}
    return I ** (q_eff / 2) * sum(
        weights.get(name, 1.0)
        * np.exp(1j * dipole_phase(np.maximum(I, cutoff_intensity(q, name, gas, wavelength)), q, name, gas, wavelength))
        for name in trajectories
    )
