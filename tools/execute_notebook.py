"""Execute the notebook with the current Python interpreter and preserve outputs.

A temporary kernel specification inside the project avoids changing user-wide
Jupyter settings. Run from any working directory using the desired interpreter.
"""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main():
    """Register a project-local kernel, execute every cell, and save after success."""
    root = Path(__file__).resolve().parents[1]
    jupyter_data = root/'tmp'/'jupyter'
    kernel = jupyter_data/'kernels'/'optical-demo'
    kernel.mkdir(parents=True, exist_ok=True)
    specification = {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                     'display_name': 'Optical demo', 'language': 'python'}
    (kernel/'kernel.json').write_text(json.dumps(specification), encoding='utf-8')
    os.environ['JUPYTER_PATH'] = str(jupyter_data)+os.pathsep+os.environ.get('JUPYTER_PATH', '')
    # Keep incidental Jupyter/IPython runtime files within this project as well.
    os.environ['JUPYTER_RUNTIME_DIR'] = str(root/'tmp'/'jupyter_runtime')
    os.environ['IPYTHONDIR'] = str(root/'tmp'/'ipython')
    path = root/'geometria_optica.ipynb'
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name='optical-demo',
                            resources={'metadata': {'path': str(root)}})
    client.execute()
    nbformat.write(notebook, path)
    print('Executed all notebook cells successfully and saved outputs.')


if __name__ == '__main__':
    main()
