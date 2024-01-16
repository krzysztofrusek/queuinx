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
              'sphinx.ext.autodoc.typehints',
              'enum_tools.autoenum',
              ]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
github_url= "https://github.com/krzysztofrusek/queuinx"

# html_theme_options = {
#     "use_repository_button": True,
#     "use_download_button": False,
# }

html_static_path = ['_static']
html_logo = "_static/images/logo.svg"

source_suffix = ['.rst', '.md', '.ipynb']
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

import queuinx as qx