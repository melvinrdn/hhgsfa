# hhgsfa

[![CI](https://github.com/melvinrdn/hhgsfa/actions/workflows/ci.yml/badge.svg)](https://github.com/melvinrdn/hhgsfa/actions/workflows/ci.yml)
[![Docs](https://github.com/melvinrdn/hhgsfa/actions/workflows/docs.yml/badge.svg)](https://melvinrdn.github.io/hhgsfa)

The single-atom response of high-order harmonic generation: strong-field approximation (SFA) dipole, spectrum and
harmonic dipole, and the analytic dipole phase of each quantum path.

**Documentation:** https://melvinrdn.github.io/hhgsfa

## Install

```bash
pip install hhgsfa            # or: uv add hhgsfa
pip install "hhgsfa[numba]"   # parallel SFA
```

## Example

Harmonic 21 of a 1030 nm driver in argon against the intensity, from the SFA and from the analytic model.

```python
import numpy as np
from hhgsfa import analytic, sfa

I = np.linspace(0.8e18, 2.4e18, 200)  # [W/m²], 0.8 to 2.4e14 W/cm²
d_sfa = sfa.harmonic_dipole(I, 21, "Ar", 1030e-9)
d_analytic = analytic.harmonic_dipole(I, 21, ("1", "2"), "Ar", 1030e-9)  # short and long paths
```

## License

MIT
