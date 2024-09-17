# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
# sys.path.insert(0, os.path.abspath('../script'))
# sys.path.insert(0, os.path.abspath('../examples'))
# os.path.abspath('../')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atpp'
copyright = '2024, Luca Santoro'
author = 'Luca Santoro'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']
autodoc_default_options = {
    'members': True,         # Include all class and module members
    'undoc-members': True,   # Include members without docstrings
    'private-members': False, # Include private members (e.g., _foo) if set to True
    'special-members': '__init__', # Include special methods like __init__
    'inherited-members': True,  # Include members inherited from parent classes
    'show-inheritance': True   # Show class inheritance diagram
}
autosummary_generate = True


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','_modules','_sources']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

