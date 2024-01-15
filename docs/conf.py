# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import date

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import queuinx
project = 'Queuinx'
copyright = f'{date.today().year}, Krzysztof Rusek'
author = 'Krzysztof Rusek'
version = queuinx.__version__

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'recommonmark',
              'sphinx.ext.autodoc.typehints']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "repository_url": "https://github.com/krzysztofrusek/queuinx",
    "use_repository_button": True,
    "use_download_button": False,
}

html_static_path = ['_static']
html_logo = "_static/images/logo.svg"
autodoc_typehints = 'none'
source_suffix = ['.rst', '.md', '.ipynb']

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"