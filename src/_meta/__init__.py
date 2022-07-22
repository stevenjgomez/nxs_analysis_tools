'''MagentroPy package metadata.'''

from os import path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__all__ = ['__project__', '__author__', '__email__', '__copyright__', '__license__', '__version__']

with open(path.join(path.abspath(path.dirname(__file__)), '../../pyproject.toml'), mode="rb") as f:
    pkg_metadata = tomllib.load(f)

__project__ = pkg_metadata['tool']['magentropy']['project_name']
__author__ = pkg_metadata['project']['authors'][0]['name']
__email__ = pkg_metadata['project']['authors'][0]['email']
__copyright__ = f"{pkg_metadata['tool']['magentropy']['copyright_year']}, {__author__}"
__license__ = pkg_metadata['tool']['magentropy']['license_name']
__version__= pkg_metadata['project']['version']
__repo_url__ = pkg_metadata['project']['urls']['Source Code']


del path, tomllib, f, pkg_metadata
