#!/usr/bin/env python3
# Produce an ultra-wide banner image from the amr_scatter snapshot — used
# as the hero figure of the README. Requires the JSON snapshot written by
#
#   ./build/examples/poisson_demo --problem amr_scatter --Nmin 4 --Nmax 7 \
#       --sigma 0.03 --output data/snapshots/amr_scatter.json
#
# Style targets a "multi-scale plasma" look:
#   - viridis colormap on V, charges visible as bright halos on a dark field
#   - AMR mesh overlaid (edges only), very visible around each charge
#   - V contour lines on top for texture
#   - red dots at charge positions
#   - ultra-wide aspect: matplotlib stretches the square domain horizontally
#     via `aspect='auto'`, so the mesh cells themselves become rectangles,
#     exactly the look of the reference banner.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "snapshots" / "amr_scatter.json"
OUT  = ROOT / "docs" / "figures" / "banner.png"


def main() -> int:
    if not SNAP.exists():
        print(f"Missing snapshot: {SNAP}")
        print("Run: ./build/examples/poisson_demo --problem amr_scatter "
              "--Nmin 4 --Nmax 7 --sigma 0.03 "
              f"--output {SNAP}")
        return 1

    with open(SNAP) as f:
        data = json.load(f)
    cells = data["cells"]
    charges = data["charges"]

    xs = np.array([c["x"] for c in cells])
    ys = np.array([c["y"] for c in cells])
    hs = np.array([c["h"] for c in cells])
    Vs = np.array([c["V"] for c in cells])

    # Interpolate V onto a regular fine grid for the background imshow and
    # the contour levels. We use `nearest` first (fills everywhere) then
    # blend with `linear` where available.
    N = 512
    xi = np.linspace(0, 1, N)
    yi = np.linspace(0, 1, N)
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    pts = np.column_stack([xs, ys])
    V_grid = griddata(pts, Vs, (Xi, Yi), method="linear", fill_value=0)
    V_near = griddata(pts, Vs, (Xi, Yi), method="nearest")
    mask = np.isnan(V_grid)
    V_grid = np.where(mask, V_near, V_grid)


    # --- Banner figure ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 3.2), dpi=200)
    fig.patch.set_facecolor("white")

    # Background: |V| on the regular grid. With alternating ±q charges the
    # tails cancel at distance (V≈0), so |V| peaks *only* at each charge
    # core and stays small between them. `viridis_r` → bright yellow
    # plateau + localised dark "caves" at every charge, exactly the look
    # of the reference banner.
    absV = np.abs(V_grid)
    vmax = float(np.percentile(absV, 99))
    im = ax.imshow(absV, origin="lower", extent=(0, 1, 0, 1),
                    cmap="viridis_r", aspect="auto",
                    norm=mcolors.Normalize(vmin=0, vmax=vmax))

    # AMR mesh: rectangles drawn in data coordinates; they stretch with
    # the axes aspect so they appear rectangular on the banner — exactly
    # the cellular look from the reference.
    rects = [Rectangle((x - h / 2, y - h / 2), h, h)
             for x, y, h in zip(xs, ys, hs)]
    mesh = PatchCollection(rects, facecolor="none",
                            edgecolor=(0, 0, 0, 0.9), linewidth=0.55)
    ax.add_collection(mesh)

    # Equipotential contours of the *signed* V — shows +/- lobe
    # structure around each charge. Density tuned to avoid clutter.
    v_span = float(np.percentile(np.abs(V_grid), 95))
    v_levels = np.linspace(-v_span, v_span, 13)
    ax.contour(Xi, Yi, V_grid, levels=v_levels, colors="black",
               linewidths=0.35, alpha=0.35)

    # Red dots at charge positions (prominent, white-edged).
    cx = np.array([c["x"] for c in charges])
    cy = np.array([c["y"] for c in charges])
    ax.scatter(cx, cy, s=75, c="#d4181c",
               edgecolor="white", linewidth=1.0, zorder=10)

    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0,
                 facecolor=fig.get_facecolor())
    plt.close(fig)

    n_by_level = {}
    for c in cells:
        n_by_level[c["level"]] = n_by_level.get(c["level"], 0) + 1

    print(f"[banner] written  →  {OUT}")
    print(f"[banner]   charges   : {len(charges)}")
    print(f"[banner]   leaves    : {len(cells)}  " +
          f"(per level: {dict(sorted(n_by_level.items()))})")
    print(f"[banner]   V range   : [{Vs.min():.2e}, {Vs.max():.2e}]")
    print(f"[banner]   SOR iter  : {data['iterations']}, "
          f"residual {data['residual']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
