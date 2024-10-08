# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
import os
import sys

# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath('..'))

# Optionally, add specific paths for scripts and examples, if needed:
# sys.path.insert(0, os.path.abspath('./atpp'))
sys.path.insert(0, os.path.abspath('../examples'))

# Print the Python path for debugging (optional)
print("Python path:", sys.path)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atpp'
copyright = '2024, Luca Santoro'
author = 'Luca Santoro'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']
# Add this to conf.py
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
    'autosummary': True  # Generate summary tables with links to each member's documentation
}


autosummary_generate = True  # This automatically generates stub .rst files for every function and class



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','_modules','_sources']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#need to mock FLIR dependencies
autodoc_mock_imports = ["fnv","tkinter","cupy"]
