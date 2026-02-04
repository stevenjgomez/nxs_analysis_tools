'''nxs-analysis-tools package metadata.'''
from setuptools_scm import get_version

# keep consistent with pyproject.toml
__project__ = 'nxs-analysis-tools'
__author__ = 'Steven J. Gomez Alvarado'
__email__ = 'stevenjgomez@ucsb.edu'
__copyright__ = f"2023-2025, {__author__}"
__license__ = 'MIT'
__version__ = get_version(root='..', relative_to=__file__)
__repo_url__ = 'https://github.com/stevenjgomez/nxs_analysis_tools'
