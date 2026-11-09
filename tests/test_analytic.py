import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, fine_structure, hbar, m_e

from hhgsfa import analytic as sc

LAM, Q, GAS = 1030e-9, 17, "Ar"
OMEGA = 2 * np.pi * c / LAM
IP = 0.5792 * 27.211386 * e  # argon [J]
C_UP = {"1": 3.17, "2": 3.17, "3": 1.54, "4": 1.54, "5": 2.40, "6": 2.40}


@pytest.mark.parametrize("trajectory", list(C_UP))
def test_cutoff_energy(trajectory):
    """At the cutoff intensity, qħω = Ip + c_up·Up."""
    I = sc.cutoff_intensity(Q, trajectory, GAS, LAM)
    up = e**2 * I / (2 * epsilon_0 * c * m_e * OMEGA**2)
    assert np.isclose(Q * hbar * OMEGA, IP + C_UP[trajectory] * up, rtol=1e-6)


@pytest.mark.parametrize("trajectory", list(C_UP))
def test_nan_below_cutoff_only(trajectory):
    I_cut = sc.cutoff_intensity(Q, trajectory, GAS, LAM)
    I = np.linspace(0, 3 * I_cut, 301)
    phase = sc.dipole_phase(I, Q, trajectory, GAS, LAM)
    assert np.isnan(phase[I < I_cut]).all() and np.isfinite(phase[I >= I_cut]).all()


@pytest.mark.parametrize("trajectory", list(C_UP))
def test_derivative_matches_finite_differences(trajectory):
    I = np.linspace(1.05, 3, 20001) * sc.cutoff_intensity(Q, trajectory, GAS, LAM)
    numeric = np.gradient(sc.dipole_phase(I, Q, trajectory, GAS, LAM), I)
    np.testing.assert_allclose(sc.d_dipole_phase_dI(I, Q, trajectory, GAS, LAM)[1:-1], numeric[1:-1], rtol=1e-5)


def test_long_path_alpha_at_high_intensity():
    """The γ term vanishes at high intensity, leaving α_2 = −α_fs λ³/(2π mₑ c³), −51.7e-14 rad cm²/W at 1030 nm."""
    alpha = -fine_structure * LAM**3 / (2 * np.pi * m_e * c**3)
    assert np.isclose(sc.d_dipole_phase_dI(1e22, Q, "2", GAS, LAM), alpha, rtol=1e-3)


def test_below_threshold_warns_and_gives_nan():
    with pytest.warns(UserWarning, match="below the ionization potential"):
        assert np.isnan(sc.dipole_phase([1e18], 13, "1", GAS, LAM)).all()


def test_harmonic_dipole_amplitude_and_frozen_phase():
    I_cut = sc.cutoff_intensity(Q, "2", GAS, LAM)
    I = np.linspace(0.1, 3, 50) * I_cut
    d = sc.harmonic_dipole(I, Q, ("2",), GAS, LAM, q_eff=4)
    np.testing.assert_allclose(np.abs(d), I**2, rtol=1e-12)
    below = I < I_cut
    np.testing.assert_allclose(np.angle(d[below] / sc.harmonic_dipole(I_cut, Q, ("2",), GAS, LAM)), 0, atol=1e-12)


def test_harmonic_dipole_weights():
    I = np.linspace(1, 3, 10) * 1e18
    d12 = sc.harmonic_dipole(I, Q, ("1", "2"), GAS, LAM, weights={"2": 0.3})
    d1, d2 = (sc.harmonic_dipole(I, Q, (n,), GAS, LAM) for n in "12")
    np.testing.assert_allclose(d12, d1 + 0.3 * d2, rtol=1e-12)
