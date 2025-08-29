# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "unit_averaging"
copyright = "2025, Vladislav Morozov"
author = "Vladislav Morozov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",  # Core autodoc support
    "sphinx.ext.napoleon",  # Google/Numpy docstring parsing
    "sphinx.ext.viewcode",  # Optional: Add links to source code
]

templates_path = ["_templates"]
exclude_patterns = []

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # Wrap examples in a box
napoleon_use_admonition_for_notes = True  # Wrap notes in a box
napoleon_use_admonition_for_references = True  # Wrap references in a box
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

napoleon_custom_sections = [
    ("Attributes", "params_style")
]  # Treat Attributes like Parameters

# Autodoc settings
autodoc_class_signature = "separated"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__",
    "ignore-module-all": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "groundwork"
# html_theme = "sphinx_celery"
html_theme = 'furo'
# html_theme = 'piccolo_theme'
# html_theme = 'sphinx_rtd_theme'