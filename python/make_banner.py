#!/usr/bin/env python3
# Produce a wide banner image from the amr_dipole snapshot — used as the
# hero figure of the README. Requires the JSON snapshot written by
#
#   ./build/examples/poisson_demo --problem amr_dipole --Nmin 4 --Nmax 7 \
#       --sigma 0.03 --output data/snapshots/amr_dipole.json
#
# The figure shows three panels side by side:
#   1. ρ(x, y) — source density (Gaussian dipole, +/-)
#   2. Adaptive mesh on top of |ρ|, revealing the AMR refinement pattern
#   3. V(x, y) — the computed potential with the mesh overlaid

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "snapshots" / "amr_dipole.json"
OUT  = ROOT / "docs" / "figures" / "banner.png"


def rects(xs, ys, hs):
    """Matplotlib Rectangle patches for cell-centered AMR cells."""
    return [Rectangle((x - h / 2, y - h / 2), h, h)
            for x, y, h in zip(xs, ys, hs)]


def draw_panel(ax, rects_, values, cmap, norm, title, *, draw_mesh=True,
               mesh_alpha=0.35, mesh_lw=0.25):
    coll = PatchCollection(rects_, cmap=cmap, norm=norm)
    coll.set_array(values)
    coll.set_edgecolor("none")
    ax.add_collection(coll)
    if draw_mesh:
        mesh = PatchCollection(rects_, facecolor="none",
                                edgecolor=(0, 0, 0, mesh_alpha),
                                linewidth=mesh_lw)
        ax.add_collection(mesh)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=title,
           xticks=[], yticks=[])
    return coll


def main() -> int:
    if not SNAP.exists():
        print(f"Missing snapshot: {SNAP}")
        print("Run: ./build/examples/poisson_demo --problem amr_dipole "
              "--Nmin 4 --Nmax 7 --sigma 0.03 "
              f"--output {SNAP}")
        return 1

    with open(SNAP) as f:
        data = json.load(f)
    cells = data["cells"]
    xs  = np.array([c["x"] for c in cells])
    ys  = np.array([c["y"] for c in cells])
    hs  = np.array([c["h"] for c in cells])
    Vs  = np.array([c["V"] for c in cells])
    rhs = np.array([c["rho"] for c in cells])
    xA, yA, xB, yB = data["xA"], data["yA"], data["xB"], data["yB"]

    # Symmetric colour ranges so the +/- symmetry of the dipole is visible.
    rho_lim = float(np.abs(rhs).max())
    V_lim   = float(np.abs(Vs).max())
    norm_rho = mcolors.TwoSlopeNorm(vmin=-rho_lim, vcenter=0, vmax=rho_lim)
    norm_V   = mcolors.TwoSlopeNorm(vmin=-V_lim,   vcenter=0, vmax=V_lim)

    # --- Banner 3-panel version (README hero) ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel 1: rho.
    draw_panel(axes[0], rects(xs, ys, hs), rhs,
                cmap="RdBu_r", norm=norm_rho,
                title="ρ(x, y)  — dipôle gaussien",
                draw_mesh=False)
    for (xc, yc, sgn, col) in [(xA, yA, "+", "#7a0b0b"),
                                (xB, yB, "−", "#0b2b7a")]:
        axes[0].annotate(sgn, xy=(xc, yc), xytext=(xc, yc + 0.08),
                          ha="center", color=col, fontsize=12,
                          fontweight="bold")

    # Panel 2: mesh overlaid on |rho| faint background.
    mag = np.abs(rhs)
    draw_panel(axes[1], rects(xs, ys, hs), mag,
                cmap="Greys",
                norm=mcolors.Normalize(vmin=0, vmax=mag.max() * 2),
                title=f"Maillage AMR  ({len(cells)} feuilles, 4 niveaux)",
                draw_mesh=True, mesh_alpha=0.95, mesh_lw=0.3)
    for (xc, yc, sgn, col) in [(xA, yA, "+", "C3"), (xB, yB, "−", "C0")]:
        axes[1].text(xc, yc, sgn, ha="center", va="center",
                      fontsize=14, fontweight="bold", color=col)

    # Panel 3: V with mesh overlay.
    im_V = draw_panel(axes[2], rects(xs, ys, hs), Vs,
                       cmap="RdBu_r", norm=norm_V,
                       title="V(x, y)  — potentiel résolu",
                       draw_mesh=True, mesh_alpha=0.25, mesh_lw=0.15)
    cb = fig.colorbar(im_V, ax=axes[2], fraction=0.045, pad=0.02, shrink=0.82)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("V", fontsize=9)

    fig.suptitle(
        "Poisson 2D  ·  quadtree AMR  ·  SOR  ·  stencil FV hétérogène",
        fontsize=14, y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight",
                 facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[banner] written  →  {OUT}")
    print(f"[banner]   V_min = {Vs.min():+.3e}  V_max = {Vs.max():+.3e}")
    print(f"[banner]   leaves per level: "
          + ", ".join(f"{lv}:{int((np.array([c['level'] for c in cells]) == lv).sum())}"
                      for lv in sorted(set(c['level'] for c in cells))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
