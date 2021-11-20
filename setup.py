
from setuptools import setup


setup(
    name="cerulean",
    version="0.0.1",
    description="A library for learning, exact inference, and constraint satisfaction using factor graphs.",
    url="https://git.collab.cra.com/projects/PROP/repos/models/browse",
    author="David Rushing Dewhurst, Joseph Campolongo, Mike Reposa",
    author_email="ddewhurst@cra.com",
    license="All rights reserved.",
    packages=["cerulean"],
    install_requires=[
        "matplotlib",
        "mypy",
        "numpy",
        "opt-einsum",
        "pandas",
        "pyro-ppl",
        "pyro-api",
        "pytest",
        "pytest-cov",
        "toml",
        "torch",
    ],

)