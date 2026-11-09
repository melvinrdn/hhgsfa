import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, h

from hhgsfa import analytic, sfa

LAM, I0 = 1030e-9, 1.5e18  # 1.5e14 W/cm²
EV_PHOTON = h * c / LAM / e


def pulse(n=None, per_cycle=128, fwhm=15e-15, span=40e-15):
    """Gaussian pulse of intensity FWHM fwhm, normalized to a peak of 1."""
    t = np.linspace(-span, span, n) if n else np.arange(-span, span, LAM / c / per_cycle)
    return t, np.exp(-2 * np.log(2) * t**2 / fwhm**2) * np.cos(2 * np.pi * c / LAM * t)


def spectrum(I=I0, **kwargs):
    first_return = kwargs.pop("first_return", True)
    t, p = pulse(**kwargs)
    return sfa.dipole_spectrum(t, sfa.dipole(t, np.sqrt(2 * I / (c * epsilon_0)) * p, "Ar", LAM, first_return))


def harmonics(t, D, q):
    """Fourier integral of D exactly at the orders q (FFT bins move with the grid)."""
    return np.exp(2j * np.pi * np.outer(q, t) * c / LAM) @ D * (t[1] - t[0])


def band(eV, S, lo, hi):
    return np.mean(np.abs(S[(eV > lo) & (eV < hi)]) ** 2)


def test_zero_field_gives_zero_dipole():
    t, _ = pulse()
    assert not sfa.dipole(t, np.zeros_like(t), "Ar", LAM).any()


def test_odd_harmonics():
    """A long pulse, so the femtosecond chirp of the long path does not smear the lines."""
    eV, S = spectrum(fwhm=100e-15, span=220e-15)
    at = lambda q: np.abs(np.interp(q * EV_PHOTON, eV, S))
    for q in (19, 21, 31, 41):
        assert at(q) > 100 * at(q - 1) and at(q) > 100 * at(q + 1)


def test_cutoff_law():
    """The plateau ends at Ip + 3.17 Up (62.8 eV here)."""
    up = e**2 * I0 / (2 * epsilon_0 * c * 9.1093837e-31 * (2 * np.pi * c / LAM) ** 2) / e
    cutoff = sfa.IP_AU["Ar"] * 27.211386 + 3.17 * up
    eV, S = spectrum(per_cycle=256)
    assert band(eV, S, 25, cutoff - 10) > 1e3 * band(eV, S, cutoff + 15, cutoff + 30)


def test_dipole_spectrum_is_harmonic_on_the_fft_grid():
    t, p = pulse()
    eV, S = sfa.dipole_spectrum(t, p)
    k = np.argmin(np.abs(eV - 17 * EV_PHOTON))
    assert np.isclose(S[k], sfa.harmonic(t, p, eV[k] / EV_PHOTON, LAM), rtol=1e-9)


def test_dipole_spectrum_axis():
    t, _ = pulse()
    eV, _ = sfa.dipole_spectrum(t, np.zeros_like(t))
    assert eV[0] == 0 and np.allclose(np.diff(eV), h / (len(t) * (t[1] - t[0])) / e)


@pytest.mark.parametrize("first_return", [True, False])
def test_harmonics_converge(first_return):
    """64 points per cycle within 1e-3 of 128 on H13-H29 (2e13 W/cm², cutoff 22 eV), 3e-4 now, 3e-2 with trapezoid A."""
    q = np.arange(13, 30, 2)
    S = []
    for n in (1500, 3000):
        t, p = pulse(n=n)
        S.append(harmonics(t, sfa.dipole(t, np.sqrt(2 * 2e17 / (c * epsilon_0)) * p, "Ar", LAM, first_return), q))
    assert np.abs(S[0] - S[1]).max() < 1e-3 * np.abs(S[1]).max()


def test_numba_matches_numpy(monkeypatch):
    if sfa.njit is None:
        pytest.skip("numba not installed")
    t, p = pulse()
    E = np.sqrt(2 * I0 / (c * epsilon_0)) * p
    D_numba = sfa.dipole(t, E, "Ar", LAM)
    monkeypatch.setattr(sfa, "njit", None)
    np.testing.assert_allclose(D_numba, sfa.dipole(t, E, "Ar", LAM), rtol=1e-9, atol=1e-12 * np.abs(D_numba).max())


def test_harmonic_of_a_harmonic_wave():
    """D = g(t)·cos(qωt), g Gaussian of width σ: the amplitude at qω is (2π)^(−1/2)·½∫g dt = ½√π·σ/√(2π)."""
    t, sigma = np.linspace(-200e-15, 200e-15, 40001), 20e-15
    D = np.exp(-((t / sigma) ** 2)) * np.cos(2 * np.pi * 17 * c / LAM * t)
    assert np.isclose(sfa.harmonic(t, D, 17, LAM), 0.5 * np.sqrt(np.pi) * sigma / np.sqrt(2 * np.pi), rtol=1e-9)


def test_harmonic_time_convention():
    """A dipole Re[A·exp(−iqωt)] under a Gaussian g gives A·∫g/2/√(2π), the time convention of analytic."""
    t, sigma, A = np.linspace(-200e-15, 200e-15, 40001), 20e-15, 0.3 - 0.8j
    D = np.real(A * np.exp(-1j * 2 * np.pi * 17 * c / LAM * t)) * np.exp(-((t / sigma) ** 2))
    assert np.isclose(sfa.harmonic(t, D, 17, LAM), A * 0.5 * np.sqrt(np.pi) * sigma / np.sqrt(2 * np.pi), rtol=1e-9)


def test_harmonic_shapes():
    t, p = pulse()
    D = np.stack([p, 2 * p])
    assert np.shape(sfa.harmonic(t, p, 17, LAM)) == ()
    assert sfa.harmonic(t, D, [15, 17, 19], LAM).shape == (2, 3)
    d = sfa.harmonic(t, p, 17, LAM)
    np.testing.assert_allclose(sfa.harmonic(t, D, 17, LAM), np.array([1, 2]) * d)


def test_nonuniform_grid_raises():
    t, p = pulse()
    t[10] += 1e-18
    with pytest.raises(ValueError, match="uniform"):
        sfa.dipole(t, p, "Ar", LAM)


def test_harmonic_above_nyquist_raises():
    t, p = pulse(per_cycle=8)
    with pytest.raises(ValueError, match="Nyquist"):
        sfa.harmonic(t, p, 17, LAM)


def test_coarse_grid_warns():
    t, p = pulse(per_cycle=48)
    with pytest.warns(UserWarning, match="too coarse"):
        sfa.dipole(t, np.sqrt(2 * I0 / (c * epsilon_0)) * p, "Ar", LAM)


def test_harmonic_dipole_is_steady():
    """The Fourier coefficient does not depend on how many steady cycles are averaged."""
    d1 = sfa.harmonic_dipole(1.5e18, [17, 21, 31], "Ar", LAM, flat=4, last=1)
    d2 = sfa.harmonic_dipole(1.5e18, [17, 21, 31], "Ar", LAM, flat=4, last=2)
    np.testing.assert_allclose(d1, d2, rtol=1e-6)


def test_harmonic_dipole_shapes():
    assert np.shape(sfa.harmonic_dipole(1.5e18, 17, "Ar", LAM)) == ()
    assert sfa.harmonic_dipole([1e18, 2e18], [17, 21], "Ar", LAM).shape == (2, 2)


def test_harmonic_dipole_paths_at_the_analytic_slopes():
    """The Fourier transform along I of d_q(I) peaks at the α of the short and long paths of the analytic model."""
    I = np.linspace(0.8e18, 2.4e18, 64)
    d = sfa.harmonic_dipole(I, 21, "Ar", LAM)
    alpha = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(I), I[1] - I[0]))
    F = np.abs(np.fft.fftshift(np.fft.fft(d * np.hanning(len(I)))))
    for trajectory in "12":
        expected = analytic.d_dipole_phase_dI(I.mean(), 21, trajectory, "Ar", LAM)
        near = np.abs(alpha - expected) < 8e-18  # within 8e-14 rad cm²/W
        assert F[near].max() > 0.3 * F.max()
