'''nxs-analysis-tools package metadata.'''

# keep consistent with pyproject.toml
__project__ = 'nxs-analysis-tools'
__author__ = 'Steven J. Gomez Alvarado'
__email__ = 'stevenjgomez@ucsb.edu'
__copyright__ = f"2023-2025, {__author__}"
__license__ = 'MIT'
__repo_url__ = 'https://github.com/stevenjgomez/nxs_analysis_tools'

try:
    import os
    from setuptools_scm import get_version
    # Try environment variable first (works for RTD or source tarballs)
    env_version = os.environ.get(
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NXS_ANALYSIS_TOOLS"
    )
    if env_version:
        __version__ = env_version
    else:
        # normal git-based version
        __version__ = get_version(root="..", relative_to=__file__)
except (ImportError, LookupError):
    # fallback hardcoded version for safety
    __version__ = "0.1.14a0"
