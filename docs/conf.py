"""Sphinx configuration of the hhgsfa documentation."""

from importlib.metadata import version as package_version

project = "hhgsfa"
author = "Melvin Redon"
copyright = "2025-2026, Melvin Redon"
release = version = package_version("hhgsfa")

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
]
exclude_patterns = ["_build", "benchmark_results.md"]  # included in benchmark.md

myst_enable_extensions = ["dollarmath", "amsmath"]
numpydoc_show_class_members = False
autodoc_typehints = "none"  # the types are in the docstrings
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 120)]
plot_rcparams = {"figure.figsize": (8, 3.4), "font.size": 10, "image.cmap": "bone"}

html_theme = "pydata_sphinx_theme"
html_title = "hhgsfa"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/melvinrdn/hhgsfa",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
