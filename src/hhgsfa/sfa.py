"""Strong-field approximation (SFA) dipole, its spectrum and its harmonics.

Lewenstein et al., Phys. Rev. A 49, 2117 (1994).
"""

import logging
import warnings

import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.typing import ArrayLike
from scipy.constants import (
    c as C_0,
    e as E_CHARGE,
    epsilon_0 as EPS_0,
    hbar as HBAR,
    m_e as M_E,
    physical_constants as _pc,
)
from scipy.integrate import cumulative_trapezoid

try:
    from numba import njit, prange
except ImportError:
    njit = None

logger = logging.getLogger(__name__)
AU_TIME = _pc["atomic unit of time"][0]
AU_EFIELD = _pc["atomic unit of electric field"][0]
HARTREE_EV = _pc["Hartree energy in eV"][0]

IP_AU: dict[str, float] = {  # ionization potentials [a.u.]
    "He": 0.9036,
    "Ne": 0.7926,
    "Ar": 0.5792,
    "Kr": 0.5145,
    "Xe": 0.4459,
}


def _integrand(dintA, dintAsq, A_birth, A_emit, E_birth, tau, inv_tau, Ip, kappa2):
    """Real amplitude and action S of the SFA integrand, scalars or arrays.

    The amplitude is d(p - A_birth) d*(p - A_emit) E_birth over |8/π √(2κ⁵)|², with the hydrogen-like transition dipole
    d(p) of Podolsky and Pauling, Phys. Rev. 34, 109 (1929).
    """
    p = dintA * inv_tau  # stationary momentum
    S = Ip * tau + 0.5 * (dintAsq - dintA * p)
    a, b = p - A_birth, p - A_emit
    x = (a * a + kappa2) * (b * b + kappa2)
    return a * b / (x * x * x) * E_birth, S


def _dipole_numpy(A, intA, intAsq, E, tau, inv_tau, weight, Ip, kappa2, D):
    """Emission times in chunks, births as i - k, the τ sum as a dot product."""
    k = np.arange(1, len(tau) + 1)
    chunk = max(1, 2**20 // len(tau))
    for i0 in range(0, len(A), chunk):
        i = np.arange(i0, min(i0 + chunk, len(A)))[:, None]
        j = i - k
        born = j >= 0
        j = np.where(born, j, 0)
        g, S = _integrand(intA[i] - intA[j], intAsq[i] - intAsq[j], A[j], A[i], born * E[j], tau, inv_tau, Ip, kappa2)
        D[i[:, 0]] = -2 * np.imag((g * np.exp(-1j * S)) @ weight)


if njit is not None:
    _integrand_nb = njit(cache=True)(_integrand)

    @njit(parallel=True, cache=True)
    def _dipole_numba(A, intA, intAsq, E, tau, inv_tau, weight, Ip, kappa2, D):
        """One emission time per thread, births summed in a compiled loop."""
        for i in prange(len(A)):
            acc = 0.0
            for k in range(min(len(tau), i)):
                j = i - k - 1
                g, S = _integrand_nb(
                    intA[i] - intA[j], intAsq[i] - intAsq[j], A[j], A[i], E[j], tau[k], inv_tau[k], Ip, kappa2
                )
                acc += g * (np.cos(S) * weight[k].imag - np.sin(S) * weight[k].real)
            D[i] = -2 * acc


def dipole(
    t: np.ndarray,
    E: np.ndarray,
    gas: str,
    wavelength: float,
    first_return: bool = True,
    eps: float = 0.0,
) -> np.ndarray:
    """SFA dipole of an atom driven by a linearly polarized field.

    Runs in parallel with numba when installed (hhgsfa[numba]).

    Parameters
    ----------
    t : ndarray
        Uniform time grid [s].
    E : ndarray
        Real driving field Re{E} on t [V/m].
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].
    first_return : bool, default True
        Keep only the short and long paths of the first return (excursions up to one cycle).
    eps : float, default 0
        Regularization ε [a.u.] of the momentum integral, (π/(ε + iτ/2))^(3/2). It can stay 0 since τ ≥ dt.

    Returns
    -------
    ndarray
        Dipole D(t) [a.u.].

    Raises
    ------
    ValueError
        If t is not uniform.

    Warns
    -----
    UserWarning
        If t is too coarse to resolve the cutoff Ip + 3.17 Up.
    """
    if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-6, atol=0):
        raise ValueError("t must be a uniform grid")
    t_au, E_au = t / AU_TIME, E / AU_EFIELD
    dt, T0 = t_au[1] - t_au[0], wavelength / C_0 / AU_TIME
    Ip = IP_AU[gas]
    kappa2 = 2 * Ip
    cutoff = Ip + 3.17 * np.abs(E_au).max() ** 2 * T0**2 / (16 * np.pi**2)  # Ip + 3.17 Up
    if np.pi / dt < cutoff:
        warnings.warn(
            f"t is too coarse to resolve the cutoff ({cutoff * HARTREE_EV:.0f} eV), "
            f"use at least {int(np.ceil(cutoff * T0 / np.pi))} points per cycle",
            stacklevel=2,
        )

    def integral(f, df):
        """Cumulative trapezoid with its end correction −dt²/12·(f′(t) − f′(t₀)), 4th order."""
        return cumulative_trapezoid(f, dx=dt, initial=0) - dt**2 / 12 * (df - df[0])

    A = -integral(E_au, np.gradient(E_au, dt, edge_order=2))
    intA = integral(A, -E_au)  # A′ = −E
    intAsq = integral(A**2, -2 * A * E_au)

    W = int(np.ceil(1.05 * T0 / dt)) if first_return else len(t)  # births per emission time
    tau = np.arange(1, W + 1) * dt
    weight = (np.pi / (eps + 0.5j * tau)) ** 1.5 * dt * 128 / np.pi**2 * kappa2**2.5  # trapezoid, 0 at both ends
    if first_return:  # 1 up to 0.95 cycle, 0 from 1.05 cycle where the second return starts
        weight *= np.sin(np.pi / 2 * np.clip((1.05 - tau / T0) / 0.1, 0, 1)) ** 2

    D = np.zeros(len(t))
    logger.debug("SFA dipole on %d points, %d births each, %s", len(t), W, "numba" if njit is not None else "numpy")
    kernel = _dipole_numba if njit is not None else _dipole_numpy
    kernel(A, intA, intAsq, E_au, tau, 1 / tau, weight, Ip, kappa2, D)
    return D


def dipole_spectrum(t: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum of a dipole, S(ω) = (2π)^(−1/2)·∫D(t)·exp(iωt)·dt, by FFT.

    The unitary angular Fourier transform with the time origin of t, as `harmonic` at every FFT frequency.

    Parameters
    ----------
    t : ndarray
        Uniform time grid [s].
    D : ndarray
        Dipole on t.

    Returns
    -------
    eV : ndarray
        Photon energy [eV].
    S : ndarray
        Complex spectrum.
    """
    omega = 2 * np.pi * rfftfreq(len(t), d=t[1] - t[0])
    S = np.conj(rfft(D)) * np.exp(1j * omega * t[0]) * (t[1] - t[0]) / np.sqrt(2 * np.pi)
    return HBAR * omega / E_CHARGE, S


def harmonic(t: np.ndarray, D: np.ndarray, q: ArrayLike, wavelength: float) -> complex | np.ndarray:
    """Complex harmonic q of a dipole, its Fourier transform (2π)^(−1/2)·∫D(t)·exp(iqωt)·dt exactly at qω.

    For the dipole of any pulse, from `dipole`. It is the amplitude of the exp(−iqωt) component, the time convention of
    `analytic`, evaluated at qω rather than read between the bins of `dipole_spectrum`. It integrates over the whole of
    t, so it grows with the pulse duration. For the steady harmonic at a fixed intensity, per period, use
    `harmonic_dipole`.

    Parameters
    ----------
    t : ndarray
        Uniform time grid [s].
    D : ndarray
        Dipole on t, along the last axis (several dipoles can be stacked).
    q : int or array_like
        Harmonic order(s).
    wavelength : float
        Central wavelength of the driver [m].

    Returns
    -------
    complex or ndarray
        Amplitude of each dipole at each harmonic, shape D.shape[:-1] + q.shape.

    Raises
    ------
    ValueError
        If qω is above the Nyquist frequency of t.
    """
    q, dt = np.asarray(q), t[1] - t[0]
    eV_q, eV_max = q.max() * HBAR * 2 * np.pi * C_0 / wavelength / E_CHARGE, np.pi * HBAR / dt / E_CHARGE
    if eV_q >= eV_max:
        raise ValueError(f"H{q.max()} ({eV_q:.1f} eV) is above the Nyquist energy of t ({eV_max:.1f} eV)")
    return np.asarray(D) @ np.exp(2j * np.pi * C_0 / wavelength * q[..., None] * t).T * dt / np.sqrt(2 * np.pi)


def harmonic_dipole(
    I: ArrayLike,
    q: ArrayLike,
    gas: str,
    wavelength: float,
    first_return: bool = True,
    ramp: float = 2,
    flat: float | None = None,
    last: int = 1,
    points_per_cycle: int | None = None,
) -> np.ndarray:
    """Steady harmonic dipole d_q(I), the Fourier coefficient of the SFA dipole in a constant field.

    Unlike `harmonic`, which takes a dipole already computed, it builds the field itself and runs `dipole` at each
    intensity, so d_q(I) is the single-atom response at that intensity, the quantity of `analytic.harmonic_dipole`.

    The field rises with a sin² ramp, stays constant for `flat` cycles, and d_q = (1/T)·∫D(t)·exp(iqωt)·dt over the last
    `last` cycles, a whole number of periods of the steady response, so the coefficient is exact without a window.

    Parameters
    ----------
    I : array_like
        Intensity [W/m²].
    q : int or array_like
        Harmonic order(s).
    gas : {"He", "Ne", "Ar", "Kr", "Xe"}
        Target gas, sets the ionization potential.
    wavelength : float
        Central wavelength of the driver [m].
    first_return : bool, default True
        Keep only the short and long paths of the first return, as in `dipole`.
    ramp : float, default 2
        Duration of the ramp [cycles].
    flat : float, optional
        Duration at constant amplitude [cycles], 3 for the first return and 20 for every return. The first return is
        steady about one cycle after the ramp. Every return converges slowly, to about 2e-2 with 10 cycles and 8e-3
        with 20, since electrons born many cycles earlier still return.
    last : int, default 1
        Number of cycles averaged, at the end of the flat part.
    points_per_cycle : int, optional
        Time steps per cycle, twice what resolves the cutoff Ip + 3.17 Up by default, at least 128.

    Returns
    -------
    ndarray
        Complex d_q [a.u.], shape I.shape + q.shape, with D(t) = Σ_q d_q·exp(−iqωt) in the steady state.
    """
    I, q = np.asarray(I, dtype=float), np.asarray(q)
    flat = (3 if first_return else 20) if flat is None else flat
    T0 = wavelength / C_0
    omega0 = 2 * np.pi / T0
    d = np.empty(I.shape + q.shape, dtype=complex)
    for k, intensity in np.ndenumerate(I):
        Up = E_CHARGE**2 * intensity / (2 * EPS_0 * C_0 * M_E * omega0**2)
        cutoff = IP_AU[gas] * HARTREE_EV * E_CHARGE + 3.17 * Up  # [J]
        n = points_per_cycle or max(128, 2 * int(np.ceil(cutoff * T0 / (np.pi * HBAR))))
        t = np.arange(round((ramp + flat) * n)) * (T0 / n)
        E = np.sqrt(2 * intensity / (C_0 * EPS_0)) * np.sin(np.pi / 2 * np.clip(t / (ramp * T0), 0, 1)) ** 2
        D = dipole(t, E * np.cos(omega0 * t), gas, wavelength, first_return)
        w = slice(len(t) - last * n, None)
        d[k] = harmonic(t[w], D[w], q, wavelength) * np.sqrt(2 * np.pi) / (last * T0)
    return d
