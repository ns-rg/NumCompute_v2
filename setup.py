# Packing Configuration for our project
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="NumCompute",
    version="2.0",
    description="A modular, production‑grade scientific computing toolkit using plain Python + NumPy",
    long_description=README,
    author="Nisarg Patel",
    packages=find_packages(),  # automatically find all packages in the project
    install_requires=["numpy"],  # Add packages accordingly when u install them
)
