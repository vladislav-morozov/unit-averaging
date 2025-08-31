# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unit Averaging"
copyright = "2025, Vladislav Morozov"
author = "Vladislav Morozov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",  # Core autodoc support
    "sphinx.ext.napoleon",  # Google/Numpy docstring parsing
    "sphinx.ext.viewcode",  # Optional: Add links to source code
    'sphinxext.opengraph',
    "sphinx_copybutton",
    "myst_parser",  # Add this
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
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__",
    "ignore-module-all": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",  # For code blocks
]

# html_theme_options = {
#     "light_css_variables": {
#         "font-stack": "Arial, sans-serif",
#         "font-stack--monospace": "Courier, monospace",
#         "font-stack--headings": "Roboto Slab, sans-serif",
#     },
# }

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#9718DC",
        "color-brand-content": "#9718DC",
        "color-admonition-background": "#B491C7",
        "color-foreground-secondary": "#545353",
        "color-background-secondary": "#DEDEDE",
        "color-api-pre-name": "#49004D",
        "color-api-name": "#9718DC",
    },
    "dark_css_variables": {
        "color-brand-primary": "gold",
        "color-brand-content": "#9718DC",
        "color-admonition-background": "gold",
        "color-api-pre-name": "#CABD46",
        "color-api-name": "gold",
        "color-foreground-secondary": "#D9D9D9",
        "color-background-border": "#313131",
        "color-brand-visited": "#fff",
        "color-admonition-background": "#313131"
    },
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/vladislav-morozov/unit-averaging/",
    "source_branch": "main",
    "source_directory": "docs/source",
}

html_title = "Unit Averaging"

pygments_style = "emacs"
pygments_dark_style = "monokai"


# Open Graph configuration
ogp_site_url = "https://vladislav-morozov.github.io/unit-averaging/"
