# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unit Averaging"
copyright = "2025, Vladislav Morozov"
author = "Vladislav Morozov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# Generate autosummary pages automatically
autosummary_generate = True
autosummary_imported_members = True

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
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
    "exclude-members": "__weakref__",
    "ignore-module-all": True,
}

# MyST
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# GitHub repo configuration
html_context = {
    "display_github": True,
    "github_user": "vladislav-morozov",
    "github_repo": "unit-averaging",
    "github_version": "main",  # or 'master', or a tag like 'v1.0'
}

# Gallery settings
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts 
    'filename_pattern': '/plot_',
    'ignore_pattern': 'germany_plot_utils.py',
    "gallery_dirs": "tutorials",  # path to where to save gallery generated output 
    "within_subsection_order": "FileNameSortKey",
    "min_reported_time": 60,
}


# Viewcode options
html_show_sourcelink = False
html_sourcelink = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#9718DC",
        "color-brand-content": "#9718DC",
        "color-admonition-background": "#C3C3C3",
        "color-foreground-secondary": "#545353",
        "color-background-secondary": "#DEDEDE",
        "color-api-pre-name": "#49004D",
        "color-api-name": "#9718DC",
    },
    "dark_css_variables": {
        "color-brand-primary": "gold",
        "color-brand-content": "#9718DC",
        "color-admonition-background": "#242424",
        "color-api-pre-name": "#CABD46",
        "color-api-name": "gold",
        "color-foreground-secondary": "#D9D9D9",
        "color-background-border": "#313131",
        "color-brand-visited": "#8830B8",
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

# MathJax
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Copy button
copybutton_exclude = ".linenos, .gp"

# Nitpick ignores for build
nitpick_ignore = [
    ("py:class", "np.ndarray"),
    ("py:class", "np.floating"),
    ("py:class", "abc.ABC"),
]


# Handling the source button
def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    try:
        submod = __import__(modname, fromlist=[""])
        obj = submod
        for part in fullname.split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    # Remap installed path to source path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_fn = os.path.relpath(fn, start=project_root)

    # If your source lives in 'src/', adjust accordingly:
    if "site-packages" in rel_fn:
        rel_fn = rel_fn.split("site-packages/")[-1]

    return_path = (
        f"https://github.com/vladislav-morozov/unit-averaging/blob/develop/"
        f"/src/{rel_fn}#L{lineno}-L{lineno + len(source) - 1}"
    )

    return return_path


html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/custom.css")  # may also be an URL
