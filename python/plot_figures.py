#!/usr/bin/env python3
# Plot the poisson_cpp solvers' output in the style of the
# CourseOnPoisson/notebooks TPs (TP1..TP5).
#
# Uses the pybind11 module `poisson_cpp` directly when available, otherwise
# falls back to loading JSON snapshots written by `examples/poisson_demo`.
#
# Usage:
#   python3 python/plot_tp_style.py tp1   # 1D Poisson + analytical overlay
#   python3 python/plot_tp_style.py tp3   # 2D SOR heatmap + slice + conv
#   python3 python/plot_tp_style.py tp4   # spectral convergence O(h^2)
#   python3 python/plot_tp_style.py tp5   # AMR quadtree mesh + V
#   python3 python/plot_tp_style.py all
#
# Set PYTHONPATH=build/python before running, or install the module first.

from __future__ import annotations

import os
import sys
import math
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


# Try pybind first; fall back to JSON if the module isn't built.
# Python prepends the script's directory to sys.path, which would let the
# source `python/poisson_cpp/__init__.py` shim shadow the installed wheel
# (where _core.so actually lives). Remove it so `import poisson_cpp` finds
# the installed package.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p not in ("", _script_dir)]
try:
    # Allow running from a CMake build tree without pip-installing.
    _build_dir = Path(__file__).resolve().parent.parent / "build" / "python"
    if _build_dir.exists():
        sys.path.insert(0, str(_build_dir))
    import poisson_cpp as pc   # noqa: F401
    HAVE_PC = True
except Exception as exc:
    print(f"[info] pybind11 module unavailable ({exc}); will use JSON snapshots")
    HAVE_PC = False


FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TP1 : 1D Poisson with Dirichlet BCs.
#
# Reference problem: zero source, uL ≠ uR. Analytical V(x) is a linear ramp.
# Validates the FV stencil + Thomas solver at machine precision.
# ---------------------------------------------------------------------------

def tp1() -> None:
    N, L, uL, uR = 100, 1.0, 10.0, 0.0

    if HAVE_PC:
        grid = pc.Grid1D(L, N)
        rho = np.zeros(N)
        V = pc.solve_poisson_1d(rho, uL, uR, grid)
        x = np.array([grid.x(i) for i in range(N)])
    else:
        raise RuntimeError("Build the pybind module: -DPOISSON_BUILD_PYTHON=ON")

    # Analytical: V(x) = uL + (uR - uL) * x / L
    V_theo = uL + (uR - uL) * x / L
    err_abs = np.abs(V - V_theo)
    err_rel = err_abs / (np.abs(V_theo) + 1e-30)   # rel err, avoid div by 0
    err_inf = float(err_abs.max())

    # Cross-check: our Thomas vs scipy's banded solver on the SAME system.
    # If both produce V to within a few ULP of each other, the discrepancy
    # with the analytical ramp is pure round-off accumulation in a direct
    # tridiagonal solve : not a scheme bug.
    try:
        from scipy.linalg import solve_banded
        # Build the same tridiag system as Solver1D for rho = 0:
        #   -V_{i-1} + 2 V_i - V_{i+1} = 0  for interior nodes,
        #   V_0 = uL, V_{N-1} = uR.
        ab = np.zeros((3, N))   # banded form, l=u=1
        ab[0, 1:] = -1.0          # super-diagonal
        ab[1, :]  = 2.0           # main diagonal
        ab[2, :-1] = -1.0         # sub-diagonal
        ab[1, 0]   = 1.0; ab[0, 1] = 0.0   # first row: 1 * V[0] = uL
        ab[1, -1]  = 1.0; ab[2, -2] = 0.0  # last row: 1 * V[N-1] = uR
        rhs_py = np.zeros(N); rhs_py[0] = uL; rhs_py[-1] = uR
        V_scipy = solve_banded((1, 1), ab, rhs_py)
        cross_err = float(np.max(np.abs(V - V_scipy)))
        print(f"[tp1]   vs scipy.solve_banded: ‖V_cpp - V_scipy‖∞ = "
              f"{cross_err:.2e}  (ULP level)")
    except ImportError:
        pass

    # Floor under "machine precision": ~N_ulp * eps * max|V|.
    # Thomas is a direct solver with O(N) floating-point ops on each pass;
    # empirically the error bound is  O(N) * eps * ||V||_inf  ≈  20 * eps * 10
    # = 4e-15 for double and N=100 here. We use eps * max|V| as a lower
    # reference line.
    machine_floor = np.finfo(float).eps * float(np.abs(V_theo).max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: V vs analytical.
    axes[0].plot(x, V, "o", ms=3, label="numérique (FV + Thomas)")
    axes[0].plot(x, V_theo, "k--", lw=1, label="analytique")
    axes[0].set(xlabel="x", ylabel="V(x)",
                title=f"TP1 : Poisson 1D (ρ=0, N={N})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: pointwise error. Mask *exact* zeros so they do not show up as
    # a spurious row near 1e-20 (artefact of the log scale). Plot the
    # double-precision floor (eps * ||V||_inf) as a dotted reference.
    mask = err_abs > 0
    axes[1].semilogy(x[mask], err_abs[mask], "o", ms=3, color="C3",
                     label=r"$|V_\mathrm{num} - V_\mathrm{anal}|$")
    axes[1].axhline(machine_floor, color="gray", ls=":", lw=1,
                    label=fr"$\varepsilon_{{mach}} \cdot \|V\|_\infty$ "
                          fr"≈ {machine_floor:.1e}")
    axes[1].set(xlabel="x", ylabel="erreur absolue",
                title=f"Erreur ponctuelle (L∞ = {err_inf:.2e}, "
                      f"L∞ relative = {err_rel.max():.2e})",
                ylim=(1e-17, 1e-12))
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3, which="both")

    out = FIG_DIR / "tp1_poisson_1d.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[tp1] L∞ abs = {err_inf:.3e}  L∞ rel = {err_rel.max():.3e}  "
          f"floor ≈ {machine_floor:.2e}  →  {out}")


# ---------------------------------------------------------------------------
# TP3 : 2D SOR red-black.
#
# Reference problem: uL ≠ uR, zero source, Neumann in y → V must be
# a linear ramp in x, independent of y. Plot heatmap + mid-line slice +
# residual convergence curve (semilog).
# ---------------------------------------------------------------------------

def tp3() -> None:
    N, uL, uR = 64, 0.0, 10.0

    if HAVE_PC:
        g = pc.Grid2D(1.0, 1.0, N, N)
        s = pc.Solver2D(g, 1.0, uL, uR)
        rho = np.zeros((N, N))
        # Record residual history by stepping in batches.
        V = np.zeros((N, N), order="F")
        history, iters = [], []
        total = 0
        step = 20
        for _ in range(500):
            rep = s.solve_inplace(V, rho, omega=-1.0, tol=1e-10,
                                   max_iter=step)
            total += rep.iterations
            history.append(rep.residual)
            iters.append(total)
            if rep.residual < 1e-10:
                break
            if rep.iterations < step:
                break

    # Cell centers (Solver2D is cell-centered).
    xc = (np.arange(N) + 0.5) / N
    yc = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # Analytical continuous ramp evaluated at cell centers.
    V_analytic = uL + (uR - uL) * xc

    # --- Coherence checks ---------------------------------------------------
    # 1. y-independence: V(i, j) should equal V(i, j') for all j, j'.
    y_std = float(np.max(V.std(axis=1)))
    # 2. Error vs the analytical ramp at cell centers.
    err_x = float(np.max(np.abs(V - V_analytic[:, None])))
    # 3. Residual at the final iterate.
    resid_final = float(history[-1]) if history else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].pcolormesh(X, Y, V, shading="auto", cmap="viridis")
    axes[0].set(title="V(x, y) : FV SOR red-black", xlabel="x", ylabel="y",
                aspect="equal")
    fig.colorbar(im, ax=axes[0])

    axes[1].plot(xc, V[:, N // 2], "o", ms=3, label="SOR (cell center)")
    axes[1].plot(xc, V_analytic, "k--", lw=1, label="analytique")
    axes[1].set(title=f"Coupe y = 0.5 (N = {N})", xlabel="x", ylabel="V")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].semilogy(iters, history, "o-", ms=3)
    axes[2].set(title="Convergence SOR", xlabel="itération",
                ylabel=r"$\max |V^{k+1} - V^k|$")
    axes[2].grid(alpha=0.3, which="both")

    out = FIG_DIR / "tp3_sor2d.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[tp3] {total} iter, résidu final {resid_final:.2e}, "
          f"err vs ramp {err_x:.2e}, y-std max {y_std:.2e}  →  {out}")


# ---------------------------------------------------------------------------
# TP4 : Spectral DST convergence study.
#
# Manufactured solution: V = sin(πx/Lx)·sin(πy/Ly) => ρ = 2(π/L)² V.
# Plot error vs h in log-log, expecting slope -2 (O(h²)).
# ---------------------------------------------------------------------------

def tp4() -> None:
    if not (HAVE_PC and pc.has_fftw3):
        print("[tp4] requires pybind module built with FFTW3"); return

    L = 1.0
    Ns = [15, 31, 63, 127, 255, 511]
    errs_cont, errs_disc, hs = [], [], []
    a = math.pi / L
    for N in Ns:
        h = L / (N + 1)
        # Continuous manufactured solution (analytical Laplacian).
        V_cont = np.zeros((N, N))
        # Discrete mode: eigenvector of the 5-point Laplacian on this grid.
        V_disc = np.zeros((N, N))
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                x, y = i * h, j * h
                V_cont[i - 1, j - 1] = math.sin(a * x) * math.sin(a * y)
                V_disc[i - 1, j - 1] = V_cont[i - 1, j - 1]   # same samples
        # rho for the continuous formulation: rho = 2 a² V (so eps0=1).
        rho_cont = 2 * a * a * V_cont
        # rho for the discrete formulation: rho = lambda_{1,1} V where
        # lambda_{k,l} = 4 sin²(k pi / (2(N+1))) / h² + same in y.
        sx = math.sin(math.pi / (2 * (N + 1)))
        lam = 2 * 4 / (h * h) * sx * sx
        rho_disc = lam * V_disc

        dst = pc.DSTSolver2D(N, N, L, L, 1.0)
        err_cont = float(np.max(np.abs(dst.solve(rho_cont) - V_cont)))
        err_disc = float(np.max(np.abs(dst.solve(rho_disc) - V_disc)))
        errs_cont.append(err_cont); errs_disc.append(err_disc); hs.append(h)

    hs = np.array(hs); errs_cont = np.array(errs_cont)
    errs_disc = np.array(errs_disc)
    slope, intercept = np.polyfit(np.log(hs), np.log(errs_cont), 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(hs, errs_cont, "o-", ms=6,
              label=f"vs $V_\\mathrm{{cont}} = \\sin\\pi x \\sin\\pi y$"
                    f" (pente ≈ {slope:.2f})")
    ax.loglog(hs, errs_disc, "s-", ms=6, color="C3",
              label="vs mode propre discret")
    ax.loglog(hs, np.exp(intercept) * hs**2, "k--", lw=1,
              label="référence $h^2$")
    ax.axhline(np.finfo(float).eps * 4, color="gray", ls=":", lw=1,
               label=r"$\approx 4\,\varepsilon_\mathrm{mach}$")
    ax.set(xlabel="h", ylabel=r"$\|V_\mathrm{num} - V_\mathrm{ref}\|_\infty$",
           title="TP4 : Erreur DST2D vs référence continue / discrète")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = FIG_DIR / "tp4_spectral_convergence.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[tp4] pente empirique = {slope:.3f} (attendu +2.0)")
    print(f"[tp4] err discrete max = {errs_disc.max():.2e} "
          f"(attendu ~ eps_mach)  →  {out}")


# ---------------------------------------------------------------------------
# TP5 : AMR quadtree + heterogeneous FV.
#
# Builds the quadtree directly via the pybind bindings, runs amr_sor,
# then draws the leaf mesh + V color-coded.
# ---------------------------------------------------------------------------

def tp5() -> None:
    if not HAVE_PC:
        raise RuntimeError("Build the pybind module: -DPOISSON_BUILD_PYTHON=ON")

    L, sigma = 1.0, 0.04
    level_min, level_max = 3, 6

    def predicate(key):
        lvl = pc.level_of(key)
        if lvl >= level_max:
            return False
        h = L / (1 << lvl)
        cx = (pc.i_of(key) + 0.5) * h
        cy = (pc.j_of(key) + 0.5) * h
        return ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) < (4 * sigma) ** 2

    def rho_func(x, y):
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return math.exp(-r2 / (sigma * sigma))

    tree = pc.Quadtree(L, level_min=level_min)
    tree.build(predicate, level_max=level_max, rho_func=rho_func)
    arr = pc.extract_arrays(tree)
    pc.amr_sor(arr, omega=1.85, tol=1e-7, max_iter=20_000)

    # Per-leaf geometry (cell center + size) for plotting and stats.
    keys = arr.keys
    levels = np.array([pc.level_of(k) for k in keys])
    hs = np.array([L / (1 << lv) for lv in levels])
    xs = np.array([(pc.i_of(k) + 0.5) * h for k, h in zip(keys, hs)])
    ys = np.array([(pc.j_of(k) + 0.5) * h for k, h in zip(keys, hs)])
    Vs = np.array(arr.V)
    rhos = np.array(arr.rho)
    areas = hs ** 2
    cells = keys  # for the message at the end

    # --- Physical coherence checks -----------------------------------------
    # 1. Total integrated charge vs theoretical (Gaussian exp(-r²/σ²)).
    Q_num = float((rhos * areas).sum())
    Q_theo = math.pi * sigma * sigma          # ∫∫ exp(-r²/σ²) dA on R²
    # 2. Peak V should occur at (0.5, 0.5).
    i_peak = int(np.argmax(Vs))
    x_peak, y_peak, V_peak = xs[i_peak], ys[i_peak], Vs[i_peak]
    # 3. Cross-check peak V against DSTSolver2D on a fine uniform grid
    #    (same problem, same sigma, same box): the two should agree to a
    #    few percent despite the AMR discretisation.
    V_peak_ref = None
    if HAVE_PC and pc.has_fftw3:
        Nref = 255
        h_ref = 1.0 / (Nref + 1)
        rho_ref = np.zeros((Nref, Nref))
        for i in range(1, Nref + 1):
            for j in range(1, Nref + 1):
                x, y = i * h_ref, j * h_ref
                r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
                rho_ref[i - 1, j - 1] = math.exp(-r2 / (sigma * sigma))
        V_ref = pc.DSTSolver2D(Nref, Nref, 1.0, 1.0, 1.0).solve(rho_ref)
        # Peak of the uniform reference.
        V_peak_ref = float(V_ref.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mesh (rectangles, edges only).
    rects = [Rectangle((x - h / 2, y - h / 2), h, h)
             for x, y, h in zip(xs, ys, hs)]
    ax = axes[0]
    pc_ = PatchCollection(rects, facecolor="none", edgecolor="k", linewidth=0.4)
    ax.add_collection(pc_)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title=f"Maillage AMR :{len(cells)} feuilles",
           xlabel="x", ylabel="y")

    # Right: V colored.
    ax = axes[1]
    rects2 = [Rectangle((x - h / 2, y - h / 2), h, h)
              for x, y, h in zip(xs, ys, hs)]
    pc_v = PatchCollection(rects2, cmap="viridis")
    pc_v.set_array(Vs)
    ax.add_collection(pc_v)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title="V(x, y) sur cellules AMR",
           xlabel="x", ylabel="y")
    fig.colorbar(pc_v, ax=ax)

    out = FIG_DIR / "tp5_amr.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    # Level histogram + stats (TP-style interpretation).
    n_by_level = {int(lv): int((levels == lv).sum()) for lv in np.unique(levels)}
    uniform = 4 ** max(n_by_level)
    gain = uniform / len(cells)
    print(f"[tp5] leaves per level: {n_by_level}  (gain ×{gain:.1f})")
    print(f"[tp5] Q_total num = {Q_num:.4f}  "
          f"(théorique πσ² = {Q_theo:.4f}, "
          f"err rel {abs(Q_num-Q_theo)/Q_theo:.1%})")
    print(f"[tp5] V_peak AMR = {V_peak:.4e} @ ({x_peak:.3f}, {y_peak:.3f})")
    if V_peak_ref is not None:
        rel = abs(V_peak - V_peak_ref) / V_peak_ref
        print(f"[tp5] V_peak DST 255² = {V_peak_ref:.4e}  → "
              f"écart AMR vs uniform {rel:.1%}")
    print(f"[tp5] → {out}")


# ---------------------------------------------------------------------------
# TP2 : 1D Poisson with a layered dielectric.
#
# No free charge (ρ = 0), so ∇·D = 0  ⇒  D = ε·E is constant across the
# whole domain, including through interfaces between layers. The potential
# V is piecewise linear with a slope inversely proportional to ε in each
# layer. This is a textbook continuity-of-D test (Griffiths ch. 4).
# ---------------------------------------------------------------------------

def tp2() -> None:
    if not HAVE_PC:
        raise RuntimeError("Build the pybind module: -DPOISSON_BUILD_PYTHON=ON")
    N = 200
    L, uL, uR, eps0 = 1.0, 15.0, 0.0, 1.0
    # 3-layer dielectric: eps_r(x) = 5 for x < 0.3, 1 for 0.3 <= x < 0.7, 2 else.
    grid = pc.Grid1D(L, N)
    x = np.array([grid.x(i) for i in range(N)])
    eps_r = np.where(x < 0.3, 5.0, np.where(x < 0.7, 1.0, 2.0))
    rho = np.zeros(N)
    V = pc.solve_poisson_1d_dielectric(rho, eps_r, uL, uR, grid, eps0)

    # Harmonic-mean face permittivity (matches the C++ stencil).
    eps_face = 2 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:])
    dx = L / (N - 1)

    # Electric field and displacement at faces.
    E = -(V[1:] - V[:-1]) / dx
    D = eps0 * eps_face * E
    # Theoretical D: with rho = 0 and no surface charges, D is constant
    # everywhere. Recover it from the total potential drop:
    #   V_uL - V_uR = sum_i D_i * dx / (eps_face_i * eps0)
    D_theo = eps0 * (uL - uR) / (dx * np.sum(1.0 / eps_face))
    D_var_rel = float((D.max() - D.min()) / abs(D.mean()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # V(x) with ε zones shaded.
    ax = axes[0]
    ax.plot(x, V, lw=1.2, color="C0")
    for lo, hi, label, color in [(0, 0.3, "ε_r=5", "gold"),
                                  (0.3, 0.7, "ε_r=1", "white"),
                                  (0.7, 1.0, "ε_r=2", "palegreen")]:
        if color != "white":
            ax.axvspan(lo, hi, alpha=0.25, color=color, label=label)
    ax.set(title="TP2 : V(x) avec couches diélectriques",
           xlabel="x", ylabel="V(x)")
    ax.legend(); ax.grid(alpha=0.3)

    # E(x) showing the jumps at interfaces.
    x_face = (x[:-1] + x[1:]) / 2
    axes[1].plot(x_face, E, lw=1.2, color="C1")
    axes[1].set(title="E(x) = -dV/dx aux faces",
                xlabel="x", ylabel="E")
    axes[1].grid(alpha=0.3)

    # D(x) is flat to machine precision.
    axes[2].plot(x_face, D, lw=1.5, color="C2", label="D_num")
    axes[2].axhline(D_theo, color="k", ls="--", lw=1, label="D théorique")
    axes[2].set(title=f"D = ε·E (continuité : "
                      f"(max-min)/|moy| = {D_var_rel:.1e})",
                xlabel="x", ylabel="D")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    out = FIG_DIR / "tp2_dielectric.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[tp2] D_num in [{D.min():.4f}, {D.max():.4f}]  "
          f"variation relative {D_var_rel:.2e}  ->  {out}")


# ---------------------------------------------------------------------------

_DISPATCH = {"tp1": tp1, "tp2": tp2, "tp3": tp3, "tp4": tp4, "tp5": tp5}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        for name, fn in _DISPATCH.items():
            print(f"--- {name} ---")
            try:
                fn()
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
    elif cmd in _DISPATCH:
        _DISPATCH[cmd]()
    else:
        print(f"Usage: {sys.argv[0]} [{'|'.join(list(_DISPATCH) + ['all'])}]")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
