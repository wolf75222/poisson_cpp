# Installation

## pip

```bash
pip install git+https://github.com/wolf75222/poisson_cpp.git
```

Le wheel est compilé localement par scikit-build-core. Prérequis :
compilateur C++20 et CMake ≥ 3.20. Eigen et nlohmann_json sont récupérés
automatiquement. FFTW3 est optionnel ; sans lui, `DSTSolver1D/2D` sont
désactivés et un `RuntimeWarning` à l'import donne la commande
d'installation pour ta plateforme.

## Google Colab

```bash
!apt-get install -y libfftw3-dev > /dev/null     # optionnel, pour le DST
!pip install -q git+https://github.com/wolf75222/poisson_cpp.git
```

Si numpy est déjà importé dans la session, évite `--force-reinstall` qui
provoque un message « Restart runtime ». Pour rafraîchir uniquement
poisson_cpp après un nouveau commit :

```bash
pip install -q --force-reinstall --no-deps git+https://github.com/wolf75222/poisson_cpp.git
```

## Build manuel

```bash
git clone https://github.com/wolf75222/poisson_cpp.git
cd poisson_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPOISSON_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build
export PYTHONPATH=$PWD/build/python
```

## Activer FFTW après coup

Si tu installes FFTW après une première install sans, force la
recompilation :

```bash
pip install --force-reinstall --no-binary poisson-cpp poisson-cpp
```

Vérifie depuis Python :

```python
import poisson_cpp as pc
print(pc.has_fftw3)
print(pc.fftw_install_hint())   # si False, donne la commande système
```
