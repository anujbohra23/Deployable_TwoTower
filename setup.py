#setup details of the project
from setuptools import setup, find_packages

setup(
    name="twotower_icd",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "tqdm",
        "pyyaml",
        "faiss-cpu",
    ],
)
