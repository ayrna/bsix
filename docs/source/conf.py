# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bsix'
copyright = '2026, Francisco Moreno Cano'
author = 'Francisco Moreno Cano'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    'sphinx.ext.autodoc',     # Generate documentation from the code
    'sphinx.ext.napoleon',    # Supports Google/NumPy style docstrings
    'sphinx.ext.viewcode',    # Adds links to the source code
    'sphinx.ext.intersphinx',
]

autodoc_default_options = {
    'exclude-members': 'set_fit_request, set_score_request, set_predict_request',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 5,
    'collapse_navigation': True, 
}

# Intersphinx mapping to link to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

import os
import sys

sys.path.insert(0, os.path.abspath('../../src/')) # Local path to bsix/src