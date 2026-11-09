# Benchmark

The SFA of hhgsfa is checked against [StrongFieldApproximation.jl](https://github.com/jagot/StrongFieldApproximation.jl)
by S. Carlström, a Julia code for the same model. Both codes run on exactly the same fields, in argon at 1030 nm.

## Summary

- With every return kept, the two codes give the same harmonics, to 2 × 10⁻⁵ at 128 points per cycle and 10⁻⁷ at 512.
- With the first return only, they differ by about 13 %, because they cut the long excursions differently. hhgsfa
  uses a smooth cut, which also converges faster.
- hhgsfa is 10 to 20 times faster with numpy alone, and about 100 times faster with numba on a single core.

## Differences

Both use the Lewenstein model, the saddle-point momentum and a hydrogen-like atom without depletion. They differ in the
numerics.

| | hhgsfa | StrongFieldApproximation.jl |
|---|---|---|
| Vector potential and its integrals | from the sampled field, 4th order | analytic |
| Sum over excursion times | trapezoid | rectangle |
| First return | smooth cut at one cycle | hard cut at one cycle |
| Parallel | yes, with numba | no |

## Figures

Spectra of a 30 fs pulse at 1.5 × 10¹⁴ W/cm². The two codes lie on top of each other.

```{image} _static/benchmark/spectra.png
:alt: Spectra of both codes for a 30 fs pulse
```

Harmonics 17, 21 and 31 against the peak intensity, amplitude and phase.

```{image} _static/benchmark/intensity_scan.png
:alt: Amplitude and phase of H17, H21 and H31 against the peak intensity, both codes
```

Accuracy against the number of points per cycle, for a 10 fs pulse.

```{image} _static/benchmark/convergence.png
:alt: Convergence of both codes and their difference against the number of points per cycle
```

Run time of the dipole.

```{image} _static/benchmark/timing.png
:alt: Run time of the dipole against the number of time points, for both codes and each backend
```

## Running the benchmark

The scripts are in `benchmarks/sfa_jl`. Install the Julia reference once, then run from the root of the repository.

```bash
julia --project=env -e 'using Pkg;
    Pkg.add(url="https://github.com/jagot/StrongFieldApproximation.jl", rev="835080b");
    Pkg.add(["ElectricFields", "Unitful", "UnitfulAtomic", "NPZ"])'
JULIA_PROJECT=env uv run --extra numba --with matplotlib python benchmarks/sfa_jl/run.py
```

## Tables

```{include} benchmark_results.md
```
