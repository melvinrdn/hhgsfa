# SFA and analytic model

The two models should agree on the phase of each path. A path whose phase grows as $\alpha_j I$ shows up as a peak at
$\alpha_j$ when the harmonic dipole is Fourier transformed along the intensity. So both dipoles of harmonic 21 are
computed over a range of intensities and transformed.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   from hhgsfa import analytic, sfa

   I = np.linspace(0.8e18, 2.4e18, 256)
   d_sfa = sfa.harmonic_dipole(I, 21, "Ar", 1030e-9)
   d_analytic = analytic.harmonic_dipole(I, 21, ("1", "2"), "Ar", 1030e-9)

   unit = 1e-18  # 1e-14 rad cm²/W in rad m²/W
   n = 4096  # zero-padding, for a smooth curve
   alpha = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, I[1] - I[0])) / unit
   fig, ax = plt.subplots()
   for d, label in ((d_sfa, "SFA"), (d_analytic, "analytic")):
       F = np.abs(np.fft.fftshift(np.fft.fft(d * np.hanning(len(I)), n)))
       ax.plot(alpha, F / F.max(), label=label)
   for path in "12":
       ax.axvline(analytic.d_dipole_phase_dI(I.mean(), 21, path, "Ar", 1030e-9) / unit, color="k", ls=":")
   ax.set(xlim=(-80, 30), xlabel="α [1e-14 rad cm²/W]", ylabel="Fourier transform along the intensity", title="H21")
   ax.legend()
   fig.tight_layout()
```

The peaks fall at the same place, near 0 for the short path and −51 × 10⁻¹⁴ rad cm²/W for the long one, the dotted
lines of the analytic model. Their heights differ because the SFA decides how much each path emits, while the analytic
model takes it as an input, `weights`.
