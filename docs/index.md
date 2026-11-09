# hhgsfa

hhgsfa computes how a single atom emits high-order harmonics. It has two models. The strong-field approximation (SFA)
integrates the electron motion in any laser field. The analytic model gives the phase of each electron path in closed
form. Both return the harmonic dipole in the same form, so they can be swapped or compared.

```python
import numpy as np
from hhgsfa import analytic, sfa

I = np.linspace(0.8e18, 2.4e18, 200)  # intensity [W/m²]
d_sfa = sfa.harmonic_dipole(I, 21, "Ar", 1030e-9)  # harmonic 21 in argon at 1030 nm
d_analytic = analytic.harmonic_dipole(I, 21, ("1", "2"), "Ar", 1030e-9)  # short and long paths
```

Start with the [quickstart](quickstart.md). The SFA is checked against
[StrongFieldApproximation.jl](https://github.com/jagot/StrongFieldApproximation.jl) in the [benchmark](benchmark.md).

```{toctree}
:maxdepth: 2
:hidden:

quickstart
user_guide/index
benchmark
api
development
```
