# -*- coding: utf-8 -*-
"""
    Setup file for autodiff.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

import sys

from pkg_resources import VersionConflict, require
import setuptools

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodiff-merlionctc",
    version="0.0.1",
    author="Jiahui Tang, Wenqi Chen, Yujie Cai",
    author_email="author@example.com",
    description="A package for autodiff",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/merlionctc/cs107-FinalProject",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
