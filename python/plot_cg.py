#!/usr/bin/env python3
# Visualisation of the Conjugate Gradient solver exposed via pybind11.
# Produces two figures in docs/figures/:
#   - cg_convergence.png : residual history CG / PCG / SOR at N=128. Shows
#                          CG's classical "superlinear cliff" (long plateau
#                          then steep drop) vs SOR's smooth geometric decay.
#   - cg_scaling.png     : iteration count and wall time vs N (32..512).
#                          Both methods scale ~O(N) in iterations for Poisson
#                          2D, but CG has a ~5× smaller constant, and the
#                          wall time is ~O(N³) for both with CG ~5× faster.
#
# Usage: PYTHONPATH=build/python python3 python/plot_cg.py

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_BUILD = Path(__file__).resolve().parent.parent / "build" / "python"
if _BUILD.exists():
    sys.path.insert(0, str(_BUILD))
import poisson_cpp as pc   # noqa: E402


FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_cg(N: int, uR: float = 10.0, tol: float = 1e-10, preconditioner: bool = False):
    g = pc.Grid2D(1.0, 1.0, N, N)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F")
    t0 = time.perf_counter()
    rep, hist = pc.solve_poisson_cg(
        V, rho, g, uL=0.0, uR=uR, tol=tol, max_iter=50_000,
        use_preconditioner=preconditioner, record_history=True)
    dt = time.perf_counter() - t0
    return V, rep, hist, dt


def run_sor_history(N: int, uR: float = 10.0, tol: float = 1e-10):
    """Run Solver2D SOR in small batches to record a residual history."""
    g = pc.Grid2D(1.0, 1.0, N, N)
    solver = pc.Solver2D(g, 1.0, 0.0, uR)
    V = np.zeros((N, N), order="F")
    rho = np.zeros((N, N), order="F")
    hist, iters = [1.0], [0]
    step = 20
    total = 0
    t0 = time.perf_counter()
    while total < 20_000:
        rep = solver.solve_inplace(V, rho, omega=-1.0, tol=tol, max_iter=step)
        total += rep.iterations
        hist.append(float(rep.residual))
        iters.append(total)
        if rep.residual < tol:
            break
        if rep.iterations < step:
            break
    dt = time.perf_counter() - t0
    return V, hist, iters, dt


# ------------------------------------------------------------------ Figure 1

def fig_convergence():
    N = 128
    V_cg,  rep_cg,  hist_cg,  t_cg  = run_cg(N)
    V_pcg, rep_pcg, hist_pcg, t_pcg = run_cg(N, preconditioner=True)
    V_sor, hist_sor, iter_sor, t_sor = run_sor_history(N)

    # Linear ramp reference — both solvers should produce V ≈ linear ramp
    xc = (np.arange(N) + 0.5) / N
    V_theo = 10.0 * xc
    err_cg  = float(np.max(np.abs(V_cg[:, N // 2] - V_theo)))
    err_sor = float(np.max(np.abs(V_sor[:, N // 2] - V_theo)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(np.arange(len(hist_cg)), hist_cg,
                "o-", ms=3, label=f"CG (iter={rep_cg.iterations}, t={t_cg*1000:.1f} ms)")
    ax.semilogy(np.arange(len(hist_pcg)), hist_pcg,
                "s-", ms=3, label=f"PCG (iter={rep_pcg.iterations}, t={t_pcg*1000:.1f} ms)")
    ax.semilogy(iter_sor, hist_sor,
                "^-", ms=3, label=f"SOR ω_opt (iter≈{iter_sor[-1]}, t={t_sor*1000:.1f} ms)")
    ax.axhline(1e-10, color="gray", ls=":", lw=0.8, label="tol = 1e-10")
    ax.set(xlabel="itération", ylabel=r"$\|r_k\| / \|b\|$",
           title=f"Convergence CG / PCG / SOR sur Poisson 2D  (N={N}, rampe 0→10)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out = FIG_DIR / "cg_convergence.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[cg] convergence →  {out}")
    print(f"[cg]   err vs linéaire: CG={err_cg:.2e}, SOR={err_sor:.2e}  "
          f"(les deux résolvent le même V à la tol)")


# ------------------------------------------------------------------ Figure 2

def fig_scaling():
    Ns = [32, 64, 128, 256, 512]
    iters_cg, iters_sor, t_cg_ms, t_sor_ms = [], [], [], []
    for N in Ns:
        _, rep, _, dt = run_cg(N, tol=1e-8)
        iters_cg.append(rep.iterations)
        t_cg_ms.append(dt * 1000)
        _, _, iter_hist, dt_sor = run_sor_history(N, tol=1e-8)
        iters_sor.append(iter_hist[-1])
        t_sor_ms.append(dt_sor * 1000)
        print(f"[cg]   N={N:4d}  CG iter={rep.iterations:5d}  "
              f"SOR iter={iter_hist[-1]:5d}  "
              f"CG t={dt*1000:7.2f} ms  SOR t={dt_sor*1000:8.2f} ms")

    fig, (axI, axT) = plt.subplots(1, 2, figsize=(13, 5))

    # Iteration count scaling.
    axI.loglog(Ns, iters_cg,  "o-", ms=6, label="CG")
    axI.loglog(Ns, iters_sor, "^-", ms=6, label="SOR ω_opt")
    # Reference slopes: O(N) for CG, O(N²) for SOR.
    Nn = np.array(Ns, dtype=float)
    ref_N  = iters_cg[0]  * (Nn / Nn[0])
    ref_N2 = iters_sor[0] * (Nn / Nn[0]) ** 2
    axI.loglog(Nn, ref_N,  "k--", lw=0.8, label="$\\propto N$")
    axI.loglog(Nn, ref_N2, "k:",  lw=0.8, label="$\\propto N^2$")
    axI.set(xlabel="N (résolution par direction)",
            ylabel="nombre d'itérations pour tol = 1e-8",
            title="Scaling du nombre d'itérations")
    axI.legend(); axI.grid(alpha=0.3, which="both")

    # Wall time scaling.
    axT.loglog(Ns, t_cg_ms,  "o-", ms=6, label="CG")
    axT.loglog(Ns, t_sor_ms, "^-", ms=6, label="SOR ω_opt")
    ref_T3 = t_cg_ms[0] * (Nn / Nn[0]) ** 3
    ref_T4 = t_sor_ms[0] * (Nn / Nn[0]) ** 4
    axT.loglog(Nn, ref_T3, "k--", lw=0.8, label="$\\propto N^3$  (CG)")
    axT.loglog(Nn, ref_T4, "k:",  lw=0.8, label="$\\propto N^4$  (SOR)")
    axT.set(xlabel="N", ylabel="temps wall (ms)",
            title="Scaling du temps total")
    axT.legend(); axT.grid(alpha=0.3, which="both")

    out = FIG_DIR / "cg_scaling.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[cg] scaling   →  {out}")


def main():
    print("--- CG convergence (N=128) ---")
    fig_convergence()
    print("--- CG scaling (N=32..512) ---")
    fig_scaling()


if __name__ == "__main__":
    main()
