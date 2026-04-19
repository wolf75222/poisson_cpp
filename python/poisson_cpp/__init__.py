"""poisson_cpp Python bindings.

Wraps the compiled extension ``_core``. Warns at import if FFTW3 is missing
(spectral solvers disabled) and returns the install command for the
detected platform via ``fftw_install_hint()``.
"""

from __future__ import annotations

import platform
import warnings
from importlib.metadata import PackageNotFoundError, version

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


if not _core.has_fftw3:
    warnings.warn(
        "poisson_cpp built without FFTW3, DSTSolver1D/2D disabled. To enable:\n\n"
        f"{fftw_install_hint()}\n",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = sorted(set(getattr(_core, "__all__", [n for n in dir(_core) if not n.startswith("_")])
                    + ["fftw_install_hint", "__version__"]))
