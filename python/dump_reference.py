"""Generate JSON reference snapshots from the Python notebooks for C++ validation.

Each snapshot contains the inputs (rho, boundary values, grid parameters) and
the reference solution produced by the Python solvers. C++ tests load these
files and compare their output against them.

Usage:
    python dump_reference.py [--outdir DATA_DIR]

By default writes to `<repo_root>/data/reference/` with filenames like
`thomas_dominant_N40.json`, `solver1d_uniform_rho_N50.json`, etc.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_OUTDIR = REPO_ROOT / "data" / "reference"


def thomas(a, b, c, d):
    """Thomas algorithm; identical to the Python notebook implementation."""
    N = len(d)
    cp = np.empty(N); dp = np.empty(N); x = np.empty(N)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, N):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < N - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_poisson_1d(rho, uL, uR, L, eps0=1.0):
    N = len(rho)
    dx = L / (N - 1)
    alpha = eps0 / dx**2
    a = np.full(N, alpha)
    b = np.full(N, -2.0 * alpha)
    c = np.full(N, alpha)
    d = -np.asarray(rho, dtype=float)
    a[0] = c[0] = 0.0; b[0] = 1.0; d[0] = uL
    a[-1] = c[-1] = 0.0; b[-1] = 1.0; d[-1] = uR
    return thomas(a, b, c, d)


def solve_poisson_1d_dielectric(rho, eps_r, uL, uR, L, eps0=1.0):
    """Variable-permittivity 1D Poisson solver (harmonic mean at faces)."""
    N = len(rho)
    dx = L / (N - 1)
    eps_face = 2.0 * eps_r[1:] * eps_r[:-1] / (eps_r[1:] + eps_r[:-1]) * eps0 / dx**2
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
    d = -np.asarray(rho, dtype=float)
    a[1:]   = eps_face
    c[:-1]  = eps_face
    for i in range(1, N - 1):
        b[i] = -(a[i] + c[i])
    a[0] = c[0] = 0.0; b[0] = 1.0; d[0] = uL
    a[-1] = c[-1] = 0.0; b[-1] = 1.0; d[-1] = uR
    return thomas(a, b, c, d)


def sor_2d(N, dx, rho, uL, uR, omega=None, tol=1e-8, max_iter=50000):
    """2D SOR reference implementation with constant eps = 1 and Dirichlet x /
    Neumann y conventions."""
    if omega is None:
        omega = 2.0 / (1.0 + np.sin(np.pi / N))
    # Uniform eps = 1, so Vw = Ve = Vn = Vs = 1/dx^2 for interior faces,
    # and Vw[0,:] = Ve[-1,:] = 2/dx^2 for the Dirichlet ghost faces.
    dx2 = dx * dx
    Ve = np.full((N, N), 1.0 / dx2);  Ve[-1, :] = 2.0 / dx2
    Vw = np.full((N, N), 1.0 / dx2);  Vw[0, :]  = 2.0 / dx2
    Vn = np.full((N, N), 1.0 / dx2);  Vn[:, -1] = 0.0  # Neumann: no ghost
    Vs = np.full((N, N), 1.0 / dx2);  Vs[:,  0] = 0.0
    Vc = Ve + Vw + Vn + Vs

    V = np.zeros((N, N))
    parity = (np.indices((N, N)).sum(axis=0) & 1).astype(bool)
    masks = (~parity, parity)
    for k in range(max_iter):
        V_old = V.copy()
        for mask in masks:
            s = np.zeros_like(V)
            s[1:,  :] += Vw[1:,  :] * V[:-1, :]
            s[:-1, :] += Ve[:-1, :] * V[1:,  :]
            s[:, 1:]  += Vs[:, 1:]  * V[:, :-1]
            s[:, :-1] += Vn[:, :-1] * V[:, 1:]
            s[0,  :]  += Vw[0,  :]  * uL
            s[-1, :]  += Ve[-1, :] * uR
            V_gs = (s + rho) / Vc
            V[mask] = (1 - omega) * V[mask] + omega * V_gs[mask]
        if np.max(np.abs(V - V_old)) < tol:
            return V, k + 1
    return V, max_iter


def dump_thomas(outdir: pathlib.Path) -> None:
    rng = np.random.default_rng(42)
    N = 40
    a = rng.random(N); b = rng.random(N) + 2.0; c = rng.random(N)
    x_theo = rng.random(N)
    A = np.diag(a[1:], -1) + np.diag(b) + np.diag(c[:-1], 1)
    d = A @ x_theo
    x = thomas(a, b, c, d)
    path = outdir / "thomas_dominant_N40.json"
    path.write_text(json.dumps({
        "N": N,
        "a": a.tolist(), "b": b.tolist(), "c": c.tolist(), "d": d.tolist(),
        "x_ref": x.tolist(),
        "x_theo": x_theo.tolist(),
        "description": "Random diagonally dominant tridiagonal, Thomas solution.",
    }, indent=2))
    print(f"wrote {path}")


def dump_solver1d_uniform(outdir: pathlib.Path) -> None:
    N = 50
    L, uL, uR = 1.0, 10.0, 0.0
    rho_val = 100.0
    rho = np.full(N, rho_val)
    V = solve_poisson_1d(rho, uL, uR, L)
    path = outdir / "solver1d_uniform_rho_N50.json"
    path.write_text(json.dumps({
        "N": N, "L": L, "uL": uL, "uR": uR, "eps0": 1.0, "rho_val": rho_val,
        "rho": rho.tolist(), "V_ref": V.tolist(),
        "description": "1D Poisson, uniform rho, Dirichlet Ua=10 Uc=0, N=50.",
    }, indent=2))
    print(f"wrote {path}")


def dump_dielectric(outdir: pathlib.Path) -> None:
    N = 40
    L, uL, uR = 1.0, 10.0, 0.0
    n_diel = 4
    eps_r = np.ones(N)
    eps_r[:n_diel] = 5.0
    eps_r[-n_diel:] = 5.0
    rho = np.zeros(N)
    V = solve_poisson_1d_dielectric(rho, eps_r, uL, uR, L)
    path = outdir / "dielectric_zero_rho_N40.json"
    path.write_text(json.dumps({
        "N": N, "L": L, "uL": uL, "uR": uR, "eps0": 1.0,
        "eps_r": eps_r.tolist(), "rho": rho.tolist(), "V_ref": V.tolist(),
        "description": "Dielectric layers, no charge, Dirichlet 10/0.",
    }, indent=2))
    print(f"wrote {path}")


def dump_sor2d(outdir: pathlib.Path) -> None:
    N = 24
    L, uL, uR = 1.0, 0.0, 1.0
    dx = L / N
    rho = np.zeros((N, N))
    V, iters = sor_2d(N, dx, rho, uL, uR, tol=1e-10)
    path = outdir / "sor2d_linear_N24.json"
    path.write_text(json.dumps({
        "N": N, "L": L, "uL": uL, "uR": uR, "iterations": iters,
        "V_ref": V.tolist(),
        "description": "SOR 2D, no charge, linear profile, N=24.",
    }, indent=2))
    print(f"wrote {path}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    dump_thomas(args.outdir)
    dump_solver1d_uniform(args.outdir)
    dump_dielectric(args.outdir)
    dump_sor2d(args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
