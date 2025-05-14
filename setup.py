from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smint',
    version='0.1.0',
    author='Jurgen Kriel',
    author_email='kriel.j@wehi.edu.au',
    description='Spatial Multi-Omics Integration (SMINT) package with enhanced segmentation capabilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JurgenKriel/SMINT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'dask>=2023.0.0',
        'distributed>=2023.0.0',
        'cellpose>=2.0.0',
        'scikit-image>=0.19.0',
        'tifffile>=2023.0.0',
        'matplotlib>=3.5.0',
        'PyYAML>=6.0',
        'opencv-python>=4.5.0',
        'Pillow>=9.0.0',
        'scipy>=1.7.0',
    ],
    extras_require={
        'r_integration': ['rpy2>=3.4.0'],
        'gui': ['tkinter'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'dev': ['pytest', 'flake8', 'black'],
    },
    entry_points={
        'console_scripts': [
            'smint-segmentation=scripts.run_segmentation:main',
            'smint-alignment=scripts.run_alignment:main',
            'smint-viewer=smint.visualization.live_scan_viewer:main',
        ],
    },
    include_package_data=True,
)
