# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versionnage [SemVer](https://semver.org/lang/fr/).

## [Unreleased]

### Changed
- Doxygen : feuille de style `docs/doxygen_custom.css` plafonne les
  images de la page principale (bannière + figures de validation) à
  720 px, évitant qu'elles débordent le conteneur sur écran large.
- Nomenclature : suppression des libellés TP1..TP5 dans les docs,
  headers, scripts et snapshots. Figures et fonctions renommées
  d'après leur contenu (`poisson_1d`, `dielectric`, `sor_2d`,
  `spectral`, `amr`).

## [0.2.0] - 2026-04-20

### Added
- Bindings Python pour la famille AMR : `Quadtree`, `Cell`, `Direction`,
  `AMRArrays`, `extract_arrays`, `writeback`, `amr_sor`, `amr_residual`,
  `AMRSORParams`/`AMRSORReport`, helpers Morton (`make_key`, `level_of`,
  `i_of`, `j_of`).
- Bindings Python pour le multigrille : `gs_smooth`, `laplacian_fv`,
  `restrict_avg`, `prolongate_const`, `prolongate_bilinear`,
  `vcycle_uniform`, `vcycle_amr_composite`, `CompositeParams`.
- Binding `solve_poisson_1d_dielectric` (variante ε(x) variable).
- Helper Python `dump_amr_snapshot(tree, path, extra=None)` pour
  persister un quadtree en JSON.
- Type stubs `.pyi` générés via pybind11-stubgen + marker `py.typed`
  (PEP 561) pour autocomplétion IDE.
- Packaging pip via scikit-build-core : `pip install
  git+https://github.com/wolf75222/poisson_cpp.git`.
- Détection plateforme + commande d'install FFTW via
  `pc.fftw_install_hint()` et `RuntimeWarning` à l'import si FFTW absent.
- Site Sphinx + Doxygen publié sur GitHub Pages :
  - Sphinx (Furo + myst-parser) : Installation, Quickstart, Examples,
    Custom V-cycle, Python API.
  - Doxygen pour le C++ (sous /cpp/).
- Workflow CI Python : 50 tests pytest exécutés après les 66 tests C++.
- Workflow Docs avec déclencheur `paths` minimal et `concurrency` annule
  les runs obsolètes.
- 50 tests pytest couvrant les bindings Python (workflows AMR + MG
  bout-en-bout, dataclasses, dump_amr_snapshot, helpers Morton).

### Changed
- `python/plot_figures.py:dielectric` et `:amr` utilisent désormais les
  bindings Python directement (au lieu d'une ré-implémentation Python ou
  d'un snapshot JSON dumpé par le CLI C++).
- C++ library compilée avec `-fPIC` pour permettre le link dans la
  shared lib pybind11.
- Fix CMake 4.x : double-définition d'`Eigen3::Eigen` quand FetchContent
  l'a déjà créé.
- CI shrinké : Ubuntu/Release uniquement (drop macOS et Debug),
  `paths-ignore` pour skip les changements docs-only.
- Docs : titres en anglais (Examples, Custom V-cycle, Python API),
  contenu en français.
- Doxygen : workflow copie maintenant `docs/figures/*.png` dans la
  sortie HTML pour que les `<img>` du README résolvent.

## [0.1.0] - 2026-04-18

Première version : C++20 library, Catch2 tests, CLI `poisson_demo`,
plot scripts Python pour les figures de validation.
