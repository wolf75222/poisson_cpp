"""poisson_cpp Python bindings.

Wraps the compiled extension ``_core``. Warns at import if FFTW3 is missing
(spectral solvers disabled) and returns the install command for the
detected platform via ``fftw_install_hint()``.
"""

from __future__ import annotations

import json
import platform
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Union

from ._core import *  # noqa: F401,F403
from . import _core as _core

try:
    __version__ = version("poisson-cpp")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"


def fftw_install_hint() -> str:
    """Return the FFTW install command for the current platform.

    DSTSolver1D/2D need FFTW3 at build time. After installing FFTW with the
    command below, reinstall poisson-cpp from source so CMake picks it up.
    """
    system = platform.system()
    if system == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                osrel = f.read().lower()
        except OSError:
            osrel = ""
        if "ubuntu" in osrel or "debian" in osrel:
            pkg = "sudo apt-get install -y libfftw3-dev"
        elif "fedora" in osrel or "rhel" in osrel or "centos" in osrel:
            pkg = "sudo dnf install -y fftw-devel"
        elif "arch" in osrel:
            pkg = "sudo pacman -S --noconfirm fftw"
        else:
            pkg = "sudo apt-get install -y libfftw3-dev   # adapt to your distro"
    elif system == "Darwin":
        pkg = "brew install fftw"
    elif system == "Windows":
        pkg = "conda install -c conda-forge fftw   # or vcpkg install fftw3"
    else:
        pkg = "conda install -c conda-forge fftw"

    return (
        f"{pkg}\n"
        f"pip install --no-binary poisson-cpp --force-reinstall poisson-cpp"
    )


def dump_amr_snapshot(
    tree: "_core.Quadtree",
    path: Union[str, Path],
    extra: Union[dict, None] = None,
) -> Path:
    """Serialize an AMR :class:`Quadtree` to a JSON file.

    The schema matches the one written by the C++ CLI ``poisson_demo
    --problem amr --output ...``: a list of leaves with their geometry
    (``key``, ``level``, ``x``, ``y``, ``h``) and per-leaf scientific
    data (``V``, ``rho``).

    Parameters
    ----------
    tree : Quadtree
        Tree whose leaves should be persisted.
    path : str or Path
        Output file. Parent directory must exist.
    extra : dict, optional
        Free-form metadata merged into the top-level JSON object (for
        example ``{"sigma": 0.04}`` to record problem parameters).

    Returns
    -------
    pathlib.Path
        Absolute path of the written file.
    """
    path = Path(path)
    cells = []
    for key, cell in tree.leaves().items():
        cx, cy = tree.cell_center(key)
        lvl = _core.level_of(key)
        cells.append({
            "key": int(key),
            "level": int(lvl),
            "x": float(cx),
            "y": float(cy),
            "h": float(tree.cell_size(lvl)),
            "V": float(cell.V),
            "rho": float(cell.rho),
        })
    payload = {
        "L": float(tree.L()),
        "level_min": int(tree.level_min()),
        "num_leaves": int(tree.num_leaves()),
        "cells": cells,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload))
    return path.resolve()


if not _core.has_fftw3:
    warnings.warn(
        "poisson_cpp built without FFTW3, DSTSolver1D/2D disabled. To enable:\n\n"
        f"{fftw_install_hint()}\n",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = sorted(set(getattr(_core, "__all__", [n for n in dir(_core) if not n.startswith("_")])
                    + ["fftw_install_hint", "dump_amr_snapshot", "__version__"]))
