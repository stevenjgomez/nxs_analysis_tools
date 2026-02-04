'''nxs-analysis-tools package metadata.'''

# keep consistent with pyproject.toml
__project__ = 'nxs-analysis-tools'
__author__ = 'Steven J. Gomez Alvarado'
__email__ = 'stevenjgomez@ucsb.edu'
__copyright__ = f"2023-2025, {__author__}"
__license__ = 'MIT'
__repo_url__ = 'https://github.com/stevenjgomez/nxs_analysis_tools'

try:
    from nxs_analysis_tools._version import version as __version__
except Exception:
    __version__ = "0.0.0"
