"""
TensorEra setup.py
==================
Install via:
    pip install -e .         # Editable (developer) install
    pip install tensorera    # Standard install (when published)
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tensorera",
    version="1.0.0",
    author="TensorEra Contributors",
    description="A productive PyTorch-like deep learning framework that works on any device",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorera/tensorera",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x"],           # NVIDIA GPU support
        "rocm": ["cupy-rocm-5-0"],          # AMD GPU (ROCm) support
        "viz": ["matplotlib", "networkx"],  # Visualization tools
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
        "full": [
            "matplotlib",
            "networkx",
            "scipy",
            "tqdm",
            "Pillow",
            "h5py",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="deep learning, neural networks, machine learning, autograd, tensors",
    entry_points={
        "console_scripts": [
            "tensorera=tensorera.__main__:main",
        ],
    },
)
