import os
import sys
import importlib
import logging
from flask import Flask, render_template, redirect, url_for

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Check if SMINT is available
SMINT_AVAILABLE = False
SMINT_VERSION = None
SMINT_MODULES = []

try:
    import smint
    SMINT_AVAILABLE = True
    SMINT_VERSION = smint.__version__
    SMINT_MODULES = smint.__all__
except ImportError:
    logger.warning("SMINT package not available")

# Check optional dependencies
DEPENDENCIES = {
    "OpenCV": False,
    "Cellpose": False,
    "Dask": False,
    "Distributed": False,
    "Dask-CUDA": False,
    "NumPy": False,
    "Pandas": False,
    "Matplotlib": False,
    "scikit-image": False,
    "tifffile": False
}

def check_dependency(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.lower()
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

# Update dependency status
DEPENDENCIES["OpenCV"] = check_dependency("OpenCV", "cv2")
DEPENDENCIES["Cellpose"] = check_dependency("Cellpose", "cellpose")
DEPENDENCIES["Dask"] = check_dependency("Dask", "dask")
DEPENDENCIES["Distributed"] = check_dependency("Distributed", "distributed")
DEPENDENCIES["Dask-CUDA"] = check_dependency("Dask-CUDA", "dask_cuda")
DEPENDENCIES["NumPy"] = check_dependency("NumPy", "numpy")
DEPENDENCIES["Pandas"] = check_dependency("Pandas", "pandas")
DEPENDENCIES["Matplotlib"] = check_dependency("Matplotlib", "matplotlib")
DEPENDENCIES["scikit-image"] = check_dependency("scikit-image", "skimage")
DEPENDENCIES["tifffile"] = check_dependency("tifffile")

@app.route('/')
def index():
    return render_template('index.html', 
                          smint_available=SMINT_AVAILABLE,
                          smint_version=SMINT_VERSION,
                          smint_modules=SMINT_MODULES,
                          dependencies=DEPENDENCIES)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)