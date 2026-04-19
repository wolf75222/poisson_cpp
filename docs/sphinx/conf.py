"""Sphinx configuration for the poisson_cpp Python documentation."""

from __future__ import annotations

import importlib.metadata

project = "poisson_cpp"
author = "Romain Despoullains"
copyright = "2026, Romain Despoullains"

try:
    release = importlib.metadata.version("poisson-cpp")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0+dev"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"poisson_cpp {release}"
html_theme_options = {
    "source_repository": "https://github.com/wolf75222/poisson_cpp",
    "source_branch": "main",
    "source_directory": "docs/sphinx/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
