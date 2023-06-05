import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
import _meta

# Minimum version, enforced by sphinx
needs_sphinx = '5.0.2'

# -- Project information -----------------------------------------------------

project = _meta.__project__
copyright = _meta.__copyright__
author = _meta.__author__

# The full version, including alpha/beta/rc tags
release = _meta.__version__

# major.minor
version = release.rpartition('.')[0]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'numpydoc',
    'myst_nb',
    'sphinx_copybutton',
    'sphinxext.opengraph',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# substitutions (should be consistent with myst_substitutions below)
rst_prolog = '''
.. |DataFrame| replace:: :class:`DataFrame <pandas.DataFrame>`
.. |Series| replace:: :class:`Series <pandas.Series>`
.. |Axes| replace:: :class:`Axes <matplotlib.axes.Axes>`
.. |Axes.plot| replace:: :meth:`Axes.plot() <matplotlib.axes.Axes.plot>`
.. |Axes.imshow| replace:: :meth:`Axes.imshow() <matplotlib.axes.Axes.imshow>`
.. |Axes.contour| replace:: :meth:`Axes.contour() <matplotlib.axes.Axes.contour>`
.. |Axes.legend| replace:: :meth:`Axes.legend() <matplotlib.axes.Axes.legend>`
.. |Figure| replace:: :class:`Figure <matplotlib.figure.Figure>`
.. |Figure.colorbar| replace:: :meth:`Figure.colorbar() <matplotlib.figure.Figure.colorbar>`
.. |Colorbar| replace:: :class:`Colorbar <matplotlib.colorbar.Colorbar>`
'''

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'furo'

html_theme_options = {
    'navigation_with_keys': True,
    'source_repository': _meta.__repo_url__,
    'source_branch': 'main',
    'source_directory': 'docs/',
}

pygments_style = 'sphinx'

pygments_dark_style = 'monokai'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# -- autodoc -----------------------------------------------------------------

autosummary_generate = True

autodoc_typehints = 'none'

autodoc_class_signature = 'separated'

autodoc_member_order = 'bysource'

autodoc_type_aliases = {
    'ReadCsvFile': 'ReadCsvFile',
    'RngSeed': 'RngSeed',
    'ArrayLike': 'ArrayLike'
}

# -- numpydoc ----------------------------------------------------------------

numpydoc_use_plots = True

numpydoc_show_class_members = True

numpydoc_attributes_as_param_list = False

numpydoc_xref_param_type = True

numpydoc_xref_aliases = {
    'ints': 'int',
    'dicts': 'dict',
    'ReadCsvFile': 'magentropy.typedefs.ReadCsvFile',
    'RngSeed': 'magentropy.typedefs.RngSeed',
    'Generator': 'numpy.random.Generator',
    'BitGenerator': 'numpy.random.BitGenerator',
    'SeedSequence': 'numpy.random.SeedSequence',
    'DataFrame': 'pandas.DataFrame',
    'DataFrameGroupBy': 'pandas.core.groupby.DataFrameGroupBy',
    'Series': 'pandas.Series',
    'Figure': 'matplotlib.figure.Figure',
    'Axes': 'matplotlib.axes.Axes',
    'Colorbar': 'matplotlib.colorbar.Colorbar',
    'Colormap': 'matplotlib.colors.Colormap'
}

numpydoc_xref_ignore = {
    'label', 'optional', 'default', 'one', 'of', 'or', 'path',
}

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'pint': ('https://pint.readthedocs.io/en/stable/', None),
}

# -- myst --------------------------------------------------------------------

myst_enable_extensions = [
    'amsmath',
    'substitution'
]

# should include all from rst_prolog above, plus additional substitutions
myst_substitutions = {
    'numba': '[`numba`](https://numba.readthedocs.io/en/stable/)',
    'pint_pandas': '[`pint_pandas`](https://github.com/hgrecco/pint-pandas)',
    'DataFrame': '{class}`DataFrame <pandas.DataFrame>`',
    'Series': '{class}`Series <pandas.Series>`',
    'Axes': '{class}`Axes <matplotlib.axes.Axes>`',
    'Axes_plot': '{meth}`Axes.plot() <matplotlib.axes.Axes.plot>`',
    'Axes_imshow': '{meth}`Axes.imshow() <matplotlib.axes.Axes.imshow>`',
    'Axes_contour': '{meth}`Axes.contour() <matplotlib.axes.Axes.contour>`',
    'Axes_legend': '{meth}`Axes.legend() <matplotlib.axes.Axes.legend>`',
    'Figure': '{class}`Figure <matplotlib.figure.Figure>`',
    'Figure_colorbar': '{meth}`Figure.colorbar() <matplotlib.figure.Figure.colorbar>`',
    'Colorbar': '{class}`Colorbar <matplotlib.colorbar.Colorbar>`',
}

myst_heading_anchors = 3

nb_execution_mode = 'cache'

nb_execution_timeout = -1

nb_execution_raise_on_error = True

nb_merge_streams = True
