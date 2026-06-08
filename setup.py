from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="numcompute_stream",
    version="2.0",
    description="A modular, streaming-compatible ensemble tree-based ML framework using plain Python + NumPy",
    long_description=README,
    author="Nisarg Patel",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
)