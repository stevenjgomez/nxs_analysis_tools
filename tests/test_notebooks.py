import os
import glob
import pytest
import matplotlib
matplotlib.use('Agg')
import nbformat
from nbclient import NotebookClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "docs", "source", "examples")
NOTEBOOKS = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.ipynb")))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=[os.path.basename(p) for p in NOTEBOOKS])
def test_notebook_execution(notebook_path):
    """
    Execute Jupyter tutorial notebooks end-to-end to ensure every cell runs without error.
    """
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        resources={'metadata': {'path': os.path.dirname(notebook_path)}}
    )
    client.execute()
