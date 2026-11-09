"""Benchmark hhgsfa.sfa against StrongFieldApproximation.jl (S. Carlström) on identical fields.

Julia builds each pulse (ElectricFields.jl, truncated Gaussian on the vector potential) and computes its dipole, hhgsfa
runs on the same sampled field. Argon, 1030 nm. Figures go to docs/_static/benchmark, tables to
docs/benchmark_results.md.

Julia setup, the reference at the commit benchmarked:
    julia --project=env -e 'using Pkg;
        Pkg.add(url="https://github.com/jagot/StrongFieldApproximation.jl", rev="835080b");
        Pkg.add(["ElectricFields", "Unitful", "UnitfulAtomic", "NPZ"])'
Run from the root of hhgsfa, with JULIA (binary) and JULIA_PROJECT=env set:
    uv run --extra numba --with matplotlib python benchmarks/sfa_jl/run.py
Julia results are cached in out_jl/, delete it to rerun them.
"""

import os
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
import numba
import numpy as np
from scipy.constants import c

from hhgsfa import sfa

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS = HERE.parents[1] / "docs"
FIGS = DOCS / "_static" / "benchmark"
T0 = 1030e-9 / c
THREADS = numba.config.NUMBA_NUM_THREADS
CONV = {"first": (128, 256, 512, 1024), "all": (128, 256, 512)}
SCAN = np.linspace(0.5e14, 2e14, 151)
CASES = (
    [(f"conv_{w}_{n}", 1.5e14, 10.0, n, w) for w, ns in CONV.items() for n in ns]
    + [(f"phys_{w}", 1.5e14, 30.0, 256, w) for w in ("first", "all")]
    + [(f"scan_{i:03d}", I, 30.0, 192, "first") for i, I in enumerate(SCAN)]
)
ODD = np.arange(15, 46, 2)  # plateau harmonics at 1.5e14 W/cm² (cutoff H52)
STYLE = {"hhgsfa": {"color": "C0", "ls": "-"}, "julia": {"color": "C1", "ls": "--"}}
WINDOW = {"first": "first return", "all": "every return"}


def run_julia():
    """Write the cases and run the Julia side, which skips cached cases."""
    (HERE / "cases.txt").write_text("".join(f"{i} {I0} {fwhm} {n} {w}\n" for i, I0, fwhm, n, w in CASES))
    subprocess.run([os.environ.get("JULIA", "julia"), str(HERE / "run_jl.jl")], check=True)


def ours(t, E, window, backend="numba", threads=THREADS):
    """Dipole of hhgsfa and its run time, with the numba or the numpy kernel."""
    numba.set_num_threads(threads)
    njit, sfa.njit = sfa.njit, (sfa.njit if backend == "numba" else None)
    try:
        start = time.perf_counter()
        D = sfa.dipole(t, E, "Ar", 1030e-9, first_return=window == "first")
        return D, time.perf_counter() - start
    finally:
        sfa.njit = njit


def err(a, b):
    """Largest difference relative to the largest value of b."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def main():
    """Run both codes on every case, write the tables and the figures."""
    run_julia()
    out = HERE / "out_jl"
    ours(np.linspace(0, 50 * T0, 2000), np.zeros(2000), "first")  # compile the numba kernel
    res = {}
    for i, _, _, _, w in CASES:
        jl = np.load(out / f"{i}.npz")
        t, E = jl["t"] * sfa.AU_TIME, jl["F"] * sfa.AU_EFIELD
        D, tn = ours(t, E, w)
        r = {"t": t, "D": D, "D_jl": jl["d"], "time_jl": float(jl["time"]), "time_numba": tn, "N": len(t)}
        if not i.startswith("scan"):
            r["time_numba1"] = ours(t, E, w, threads=1)[1]
            r["time_numpy"] = ours(t, E, w, backend="numpy")[1]
        res[i] = r
    sign = np.sign(np.dot(res["phys_all"]["D"], res["phys_all"]["D_jl"]))  # dipole sign convention of each code
    for r in res.values():
        r["D_jl"] = sign * r["D_jl"]
        r["S"], r["S_jl"] = (sfa.harmonic(r["t"], r[k], ODD, 1030e-9) for k in ("D", "D_jl"))

    FIGS.mkdir(parents=True, exist_ok=True)
    lines = [f"Julia compile time {float(np.load(out / 'compile.npz')['time']):.1f} s (excluded below).\n"]
    lines += accuracy(res)
    lines += timing(res)
    lines += physics(res)
    (DOCS / "benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    spectra(res)
    scan(res)


def accuracy(res):
    """Error of H15-H45 against the finest grid of the same code, and between the two codes."""
    header = "| window | points per cycle | hhgsfa, own convergence | julia, own convergence | between codes "
    lines = [header + "| between codes, phase [rad] |", "|---|---|---|---|---|---|"]
    conv = {}
    for w, ns in CONV.items():
        ref = res[f"conv_{w}_{ns[-1]}"]
        for n in ns:
            r = res[f"conv_{w}_{n}"]
            big = np.abs(r["S_jl"]) > 1e-3 * np.abs(r["S_jl"]).max()
            conv[w, n] = (err(r["S"], ref["S"]), err(r["S_jl"], ref["S_jl"]), err(r["S"], r["S_jl"]))
            phase = np.max(np.abs(np.angle(r["S"][big] / r["S_jl"][big])))
            own = [f"{e:.1e}" if n != ns[-1] else "reference" for e in conv[w, n][:2]]
            lines.append(f"| {WINDOW[w]} | {n} | {own[0]} | {own[1]} | {conv[w, n][2]:.1e} | {phase:.1e} |")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for w, mark in (("first", "o"), ("all", "s")):
        ns = CONV[w][:-1]
        for k, code in enumerate(("hhgsfa", "julia")):
            ax[0].loglog(ns, [conv[w, n][k] for n in ns], marker=mark, **STYLE[code], label=f"{code}, {WINDOW[w]}")
        ax[1].loglog(CONV[w], [conv[w, n][2] for n in CONV[w]], marker=mark, color="k", label=WINDOW[w])
    ax[0].set(xlabel="points per cycle", ylabel="error on H15-H45", title="against the finest grid of the same code")
    ax[1].set(xlabel="points per cycle", ylabel="difference on H15-H45", title="hhgsfa against julia")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "convergence.png", dpi=150)
    return ["\n### Accuracy\n", *lines, ""]


def timing(res):
    """Run time of the dipole for each backend."""
    header = "| case | time points | julia, 1 thread [s] | hhgsfa numpy [s] | hhgsfa numba, 1 thread [s] "
    lines = [header + f"| hhgsfa numba, {THREADS} threads [s] |", "|---|---|---|---|---|---|"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, w in zip(ax, ("first", "all"), strict=True):
        cases = [r for i, r in res.items() if i.startswith(f"conv_{w}")]
        N = [r["N"] for r in cases]
        for key, label, mark in (
            ("time_jl", "julia, 1 thread", "s"),
            ("time_numpy", "hhgsfa numpy", "o"),
            ("time_numba1", "hhgsfa numba, 1 thread", "^"),
            ("time_numba", f"hhgsfa numba, {THREADS} threads", "v"),
        ):
            a.loglog(N, [r[key] for r in cases], marker=mark, label=label)
        a.set(xlabel="time points", title=f"{WINDOW[w]}, 10 fs pulse")
        for r, n in zip(cases, CONV[w], strict=True):
            j = r["time_jl"]
            ours_ = " | ".join(f"{r[k]:.3f} (×{j / r[k]:.0f})" for k in ("time_numpy", "time_numba1", "time_numba"))
            lines.append(f"| 10 fs, {WINDOW[w]}, {n} pts/cycle | {r['N']} | {j:.2f} | {ours_} |")
    ax[0].set(ylabel="run time of the dipole [s]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "timing.png", dpi=150)
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""  # Linux only
    models = [line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")]
    cpu = models[0] if models else platform.processor() or platform.machine()
    return ["\n### Run time\n", f"Measured on {cpu}, {os.cpu_count()} logical cores.\n", *lines, ""]


def physics(res):
    """Contrast of the near-threshold harmonics in a 30 fs pulse, in both codes."""
    lines = ["| window | code | H17 | H19 | H21 | H31 | H41 |", "|---|---|---|---|---|---|---|"]
    for w in ("first", "all"):
        r = res[f"phys_{w}"]
        for code, key in (("hhgsfa", "D"), ("julia", "D_jl")):
            a = np.abs(sfa.harmonic(r["t"], r[key], np.arange(16, 43), 1030e-9))
            contrast = [min(a[q - 16] / a[q - 17], a[q - 16] / a[q - 15]) for q in (17, 19, 21, 31, 41)]
            lines.append(f"| {WINDOW[w]} | {code} | " + " | ".join(f"{x:.1f}" for x in contrast) + " |")
    caption = "Amplitude of each harmonic over the larger of its two even neighbors.\n"
    return ["\n### Harmonic contrast, 30 fs at 1.5e14 W/cm²\n", caption, *lines, ""]


def spectra(res):
    """Spectra of both codes for the 30 fs pulse."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for a, w in zip(ax, ("first", "all"), strict=True):
        r = res[f"phys_{w}"]
        q = np.fft.rfftfreq(r["N"], r["t"][1] - r["t"][0]) * T0
        band = (q > 9) & (q < 65)
        for code, key in (("hhgsfa", "D"), ("julia", "D_jl")):
            P = np.abs(np.fft.rfft(r[key])) ** 2
            a.semilogy(q[band], P[band], lw=0.8, **STYLE[code], label=code)
        a.set(xlim=(9, 65), ylim=(1e-6 * P[band].max(), 3 * P[band].max()), ylabel="|S|² [arb. u.]")
        a.set_title(f"30 fs, 1.5e14 W/cm², {WINDOW[w]}")
        for k in range(11, 65, 2):
            a.axvline(k, color="0.88", lw=0.5, zorder=0)
        a.legend(fontsize=8)
    ax[1].set(xlabel="harmonic order")
    fig.tight_layout()
    fig.savefig(FIGS / "spectra.png", dpi=150)


def scan(res):
    """Amplitude and phase of H17, H21 and H31 against the peak intensity, both codes."""
    rows = [res[f"scan_{i:03d}"] for i in range(len(SCAN))]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for col, q in enumerate((17, 21, 31)):
        for code, key in (("hhgsfa", "D"), ("julia", "D_jl")):
            S = np.array([sfa.harmonic(r["t"], r[key], q, 1030e-9) for r in rows])
            ax[0, col].semilogy(SCAN / 1e14, np.abs(S), **STYLE[code], label=code)
            ax[1, col].plot(SCAN / 1e14, np.unwrap(np.angle(S)), **STYLE[code], label=code)
        ax[0, col].set(title=f"H{q}, 30 fs, first return", ylabel="|S(qω)| [arb. u.]")
        ax[1, col].set(xlabel="peak intensity [1e14 W/cm²]", ylabel="arg S(qω) [rad]")
        ax[0, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "intensity_scan.png", dpi=150)


if __name__ == "__main__":
    main()
